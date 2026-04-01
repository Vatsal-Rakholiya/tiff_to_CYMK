import io
import os
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
import tifffile
from PIL import Image


st.set_page_config(page_title="CMYK TIFF Separation Dashboard", layout="wide")


# =========================
# Core pipeline functions
# Based on the uploaded merged pipeline.
# =========================
def normalize_to_uint8(arr):
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return (arr / 257.0).round().astype(np.uint8)
    raise ValueError(f"Unsupported dtype: {arr.dtype}")


def extract_cmyk(arr):
    """
    Convert common TIFF layouts to H x W x 4 CMYK.
    Supported:
    - H x W x 4
    - 4 x H x W
    - H x W x 5  (drops 5th channel)
    - 5 x H x W  (drops 5th channel)
    """
    arr = normalize_to_uint8(arr)

    if arr.ndim != 3:
        raise ValueError(f"Unsupported TIFF shape: {arr.shape}")

    if arr.shape[2] in (4, 5):
        return arr[:, :, :4]

    if arr.shape[0] in (4, 5):
        return np.transpose(arr[:4, :, :], (1, 2, 0))

    raise ValueError(f"Cannot interpret TIFF as CMYK. Shape: {arr.shape}")


def save_cmyk_tiff_bytes(cmyk_array):
    buf = io.BytesIO()
    tifffile.imwrite(
        buf,
        cmyk_array.astype(np.uint8),
        photometric="separated",
        planarconfig="contig",
        compression="lzw",
        metadata=None,
    )
    buf.seek(0)
    return buf.getvalue()


def save_preview_tiff_bytes(plate, white_threshold=245):
    preview = 255 - plate
    preview[preview >= white_threshold] = 255

    buf = io.BytesIO()
    tifffile.imwrite(
        buf,
        preview.astype(np.uint8),
        photometric="minisblack",
        compression="lzw",
        metadata=None,
    )
    buf.seek(0)
    return buf.getvalue(), preview


def read_uploaded_tiff(uploaded_file):
    uploaded_file.seek(0)
    with tifffile.TiffFile(uploaded_file) as tif:
        page = tif.pages[0]
        arr = page.asarray()
        info = {
            "shape": page.shape,
            "dtype": str(page.dtype),
            "photometric": str(page.photometric),
            "compression": str(page.compression),
        }
    return arr, info


def create_separations(cmyk):
    h, w, _ = cmyk.shape
    c = cmyk[:, :, 0]
    m = cmyk[:, :, 1]
    y = cmyk[:, :, 2]
    k = cmyk[:, :, 3]

    zeros = np.zeros((h, w), dtype=np.uint8)

    cyan_only = np.stack([c, zeros, zeros, zeros], axis=2)
    magenta_only = np.stack([zeros, m, zeros, zeros], axis=2)
    yellow_only = np.stack([zeros, zeros, y, zeros], axis=2)
    black_only = np.stack([zeros, zeros, zeros, k], axis=2)

    return {
        "cyan": cyan_only,
        "magenta": magenta_only,
        "yellow": yellow_only,
        "black": black_only,
    }


def pil_from_gray(gray_array):
    return Image.fromarray(gray_array, mode="L")


def build_zip(original_stem, separation_files, preview_files):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for color, data in separation_files.items():
            zf.writestr(f"{original_stem}_{color}.tif", data)
        for color, data in preview_files.items():
            zf.writestr(f"{original_stem}_{color}_plate_preview.tif", data)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# =========================
# UI
# =========================
st.title("CMYK TIFF Separation Dashboard")
st.caption("Upload one CMYK TIFF, generate 4 separated TIFFs and 4 Corel-style plate preview TIFFs.")

with st.sidebar:
    st.header("Settings")
    white_threshold = st.slider(
        "White threshold for plate preview",
        min_value=0,
        max_value=255,
        value=245,
        help="Higher values force more light pixels to pure white in the preview output.",
    )
    save_to_disk = st.checkbox("Also save files to a local output folder", value=False)
    output_folder = st.text_input("Local output folder", value="output")

uploaded_file = st.file_uploader("Upload CMYK TIFF", type=["tif", "tiff"])

if uploaded_file is not None:
    original_stem = Path(uploaded_file.name).stem

    try:
        arr, info = read_uploaded_tiff(uploaded_file)
        cmyk = extract_cmyk(arr)

        st.success("TIFF loaded successfully.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Shape", str(info["shape"]))
        col2.metric("Dtype", info["dtype"])
        col3.metric("Photometric", info["photometric"])
        col4.metric("Compression", info["compression"])

        if st.button("Run full CMYK pipeline", type="primary"):
            separations = create_separations(cmyk)

            separation_bytes = {}
            preview_bytes = {}
            preview_images = {}

            for color, sep_array in separations.items():
                separation_bytes[color] = save_cmyk_tiff_bytes(sep_array)

                channel_idx = {
                    "cyan": 0,
                    "magenta": 1,
                    "yellow": 2,
                    "black": 3,
                }[color]

                plate = sep_array[:, :, channel_idx].astype(np.uint8)
                preview_tiff, preview_gray = save_preview_tiff_bytes(
                    plate, white_threshold=white_threshold
                )
                preview_bytes[color] = preview_tiff
                preview_images[color] = preview_gray

            if save_to_disk:
                os.makedirs(output_folder, exist_ok=True)
                for color, data in separation_bytes.items():
                    with open(os.path.join(output_folder, f"{original_stem}_{color}.tif"), "wb") as f:
                        f.write(data)
                for color, data in preview_bytes.items():
                    with open(
                        os.path.join(output_folder, f"{original_stem}_{color}_plate_preview.tif"),
                        "wb",
                    ) as f:
                        f.write(data)

            st.subheader("Plate preview")
            preview_cols = st.columns(4)
            for idx, color in enumerate(["cyan", "magenta", "yellow", "black"]):
                with preview_cols[idx]:
                    st.markdown(f"**{color.title()}**")
                    st.image(pil_from_gray(preview_images[color]), use_container_width=True)

            st.subheader("Downloads")
            for color in ["cyan", "magenta", "yellow", "black"]:
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(
                        label=f"Download {color.title()} separation TIFF",
                        data=separation_bytes[color],
                        file_name=f"{original_stem}_{color}.tif",
                        mime="image/tiff",
                        key=f"sep_{color}",
                    )
                with c2:
                    st.download_button(
                        label=f"Download {color.title()} plate preview TIFF",
                        data=preview_bytes[color],
                        file_name=f"{original_stem}_{color}_plate_preview.tif",
                        mime="image/tiff",
                        key=f"prev_{color}",
                    )

            zip_bytes = build_zip(original_stem, separation_bytes, preview_bytes)
            st.download_button(
                label="Download all outputs as ZIP",
                data=zip_bytes,
                file_name=f"{original_stem}_cmyk_outputs.zip",
                mime="application/zip",
            )

            st.success("Pipeline completed.")
            if save_to_disk:
                st.info(f"Files also saved to: {os.path.abspath(output_folder)}")

    except Exception as e:
        st.error(f"Failed to process TIFF: {e}")
else:
    st.info("Upload a CMYK TIFF to begin.")
