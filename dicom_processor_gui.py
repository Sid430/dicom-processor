import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog, QLineEdit, \
    QMessageBox, QProgressBar, QMainWindow

FACE_MAX_VALUE = 50
FACE_MIN_VALUE = -125

AIR_THRESHOLD = -800
KERNEL_SIZE = 50
ERROR = ""
import pydicom
import numpy as np
import os
class DicomProcessor:
    def __init__(self):
        self.error = ""

    def is_dicom(self, file_path):
        try:
            ds = pydicom.dcmread(file_path, force=True)
            ds.decompress()
            if self.checkCTmeta(ds) == 0:
                return False
            return True
        except Exception:
            return False

    def list_dicom_directories(self, root_dir):
        dicom_dirs = set()
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if self.is_dicom(file_path):
                    dicom_dirs.add(root)
                else:
                    break
        return list(dicom_dirs)

    def load_scan(self, path):
        p = Path(path)
        if p.is_file():
            slices = pydicom.dcmread(str(p), force=True)
        return slices

    def get_pixels_hu(self, slices):
        image = slices.pixel_array.astype(np.int16)
        image[image == -2000] = 0
        intercept = slices.RescaleIntercept
        slope = slices.RescaleSlope
        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)
        image += np.int16(intercept)
        return np.array(image, dtype=np.int16)

    def binarize_volume(self, volume, air_hu=-800):
        binary_volume = np.zeros_like(volume, dtype=np.uint8)
        binary_volume[volume <= air_hu] = 1
        return binary_volume

    def largest_connected_component(self, binary_image):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        largest_component_image = np.zeros(labels.shape, dtype=np.uint8)
        largest_component_image[labels == largest_component_index] = 1
        return largest_component_image

    def get_largest_component_volume(self, volume):
        processed_volume = self.largest_connected_component(volume)
        return processed_volume

    def dilate_volume(self, volume, kernel_size=KERNEL_SIZE):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated_volume = cv2.dilate(volume.astype(np.uint8), kernel)
        return dilated_volume

    def apply_mask_and_get_values(self, image_volume, mask_volume):
        masked_volume = image_volume * mask_volume
        unique_values = np.unique(masked_volume)
        unique_values = unique_values[(unique_values > -125) & (unique_values < 50)]
        return unique_values.tolist()

    def apply_random_values_optimized(self, pixels_hu, dilated_volume, unique_values_list):
        new_volume = np.copy(pixels_hu)
        random_indices = np.random.choice(len(unique_values_list), size=np.sum(dilated_volume))
        random_values = np.array(unique_values_list)[random_indices]
        new_volume[dilated_volume == 1] = random_values
        return new_volume

    def person_names_callback(self, ds, elem):
        if elem.VR == "PN":
            elem.value = "anonymous"

    def curves_callback(self, ds, elem):
        if elem.tag.group & 0xFF00 == 0x5000:
            del ds[elem.tag]

    def is_substring_in_list(self, substring, string_list):
        return any(substring in string for string in string_list)

    def checkCTmeta(self, ds):
        try:
            try:
                modality = ds[0x08, 0x60].value
            except Exception as e:
                modality = ds.Modality

            modality = [modality] if isinstance(modality, str) else modality
            modality = list(map(lambda x: x.lower().replace(' ', ''), modality))
            check = ["ct", "computedtomography", "ctprotocal"]
            status1 = any(self.is_substring_in_list(c, modality) for c in check)
            try:
                imageType = ds[0x08, 0x08].value
            except Exception as e:
                imageType = ds.ImageType

            imageType = [imageType] if isinstance(imageType, str) else imageType
            imageType = list(map(lambda x: x.lower().replace(' ', ''), imageType))
            check = ["original", "primary", "axial"]
            status2 = any(self.is_substring_in_list(c, imageType) for c in check)
            try:
                studyDes = ds[0x08, 0x1030].value
            except Exception as e:
                studyDes = ds.StudyDescription
            studyDes = [studyDes] if isinstance(studyDes, str) else studyDes
            studyDes = list(map(lambda x: x.lower().replace(' ', ''), studyDes))
            check = ["head", "brain", "skull"]
            status3 = any(self.is_substring_in_list(c, studyDes) for c in check)

            return int(status1 and status2 and status3)
        except Exception as e:
            self.error = str(e)
        return 0

    def save_new_dicom_files(self, original_dir, out_dir, replacer='face', id='GWTG_ID', name='Processed for GWTG'):
        dicom_files = [f for f in os.listdir(original_dir) if self.is_dicom(os.path.join(original_dir, f))]

        try:
            dicom_files.sort(
                key=lambda x: int(pydicom.dcmread(os.path.join(original_dir, x), force=True).InstanceNumber))
        except Exception as e:
            self.error = str(e)

        errors = []

        for i, dicom_file in enumerate(dicom_files, start=1):
            try:
                ds = self.load_scan(os.path.join(original_dir, dicom_file))

                pixels_hu = self.get_pixels_hu(ds)

                binarized_volume = self.binarize_volume(pixels_hu)
                processed_volume = self.get_largest_component_volume(binarized_volume)

                dilated_volume = self.dilate_volume(processed_volume)

                if replacer == 'face':
                    unique_values_list = self.apply_mask_and_get_values(pixels_hu, dilated_volume - processed_volume)
                elif replacer == 'air':
                    unique_values_list = [0]
                else:
                    try:
                        replacer = int(replacer)
                        unique_values_list = [replacer]
                    except:
                        unique_values_list = self.apply_mask_and_get_values(pixels_hu,
                                                                            dilated_volume - processed_volume)

                new_volume = self.apply_random_values_optimized(pixels_hu, dilated_volume, unique_values_list)

                ds.decompress()
                ds.remove_private_tags()

                if "OtherPatientIDs" in ds:
                    delattr(ds, "OtherPatientIDs")
                if "OtherPatientIDsSequence" in ds:
                    del ds.OtherPatientIDsSequence
                ds.walk(self.person_names_callback)
                ds.walk(self.curves_callback)

                ANONYMOUS = "Processed GWTG"
                import time
                today = time.strftime("%Y%m%d")
                ds[0x08, 0x50].value = ANONYMOUS
                ds[0x10, 0x10].value = name
                ds[0x10, 0x20].value = id
                ds[0x10, 0x30].value = today

                if 'AccessionNumber' in ds:
                    ds.data_element('AccessionNumber').value = ANONYMOUS
                if 'PatientName' in ds:
                    ds.data_element('PatientName').value = name
                if 'PatientID' in ds:
                    ds.data_element('PatientID').value = id
                if 'PatientBirthDate' in ds:
                    ds.data_element('PatientBirthDate').value = today

                try:
                    ds[0x10, 0x1040].value = ANONYMOUS
                except Exception as e:
                    self.error = str(e)
                if 'PatientAddress' in ds:
                    ds.data_element('PatientAddress').value = ANONYMOUS

                try:
                    ds[0x10, 0x2154].value = ANONYMOUS
                except Exception as e:
                    self.error = str(e)
                if 'PatientTelephoneNumbers' in ds:
                    ds.data_element('PatientTelephoneNumbers').value = ANONYMOUS

                new_slice = (new_volume - ds.RescaleIntercept) / ds.RescaleSlope
                ds.PixelData = new_slice.astype(np.int16).tobytes()

                new_file_name = f"{id}_{i:05d}.dcm"
                final_file_path = os.path.join(out_dir, new_file_name)
                ds.save_as(final_file_path)
            except Exception as e:
                self.error = str(e)
                errors.append((dicom_file, str(e)))

        if errors:
            with open(os.path.join(out_dir, 'log.txt'), 'w') as error_file:
                for dicom_file, error in errors:
                    error_file.write(f"File: {dicom_file}, Error: {error}\n")

        return errors

    # Load the DICOM series
    dicom_directory = "path/to/dicom/files"
    def drown_volume(self, in_path, out_path, replacer='face', id='GWTG_ID', name='Processed for GWTG'):
        try:
            for root, dirs, files in os.walk(in_path):
                relative_path = os.path.relpath(root, in_path)
                out_dir = os.path.join(out_path, relative_path)

                dicom_files = [f for f in files if self.is_dicom(os.path.join(root, f))]
                if dicom_files:
                    os.makedirs(out_dir, exist_ok=True)
                    self.save_new_dicom_files(root, out_dir, replacer, id, name)
        except Exception as e:
            self.error = str(e)
            return 0
        return 1


class RenameFoldersThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(str)

    def __init__(self, dicom_folder, excel_path, out_path):
        super().__init__()
        self.dicom_folder = dicom_folder
        self.excel_path = excel_path
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_path = os.path.join(out_path, f'Processed for GWTG_{current_time}')
        self.processor = DicomProcessor()

    def run(self):
        os.makedirs(self.out_path, exist_ok=True)
        log_path = os.path.join(self.out_path, 'log.txt')
        with open(log_path, 'w') as log_file:
            try:
                df = pd.read_excel(self.excel_path)
                if 'Accession_number' not in df.columns or 'GWTG_ID' not in df.columns:
                    self.finished_signal.emit("Error: Excel file must contain 'Accession_number' and 'GWTG_ID' columns.")
                    return

                df['Accession_number'] = df['Accession_number'].astype(str).str.strip()
                df['GWTG_ID'] = df['GWTG_ID'].astype(str).str.strip()

                id_mapping = dict(zip(df['Accession_number'], df['GWTG_ID']))
                dicom_folders = [d for d in os.listdir(self.dicom_folder) if
                                 os.path.isdir(os.path.join(self.dicom_folder, d))]
                total_folders = len(dicom_folders)
                for i, foldername in enumerate(dicom_folders):
                    stripped_foldername = foldername.strip()
                    src_folder = os.path.join(self.dicom_folder, foldername)
                    try:
                        if stripped_foldername in id_mapping:
                            dst_folder = os.path.join(self.out_path, id_mapping[stripped_foldername])
                            result = self.processor.drown_volume(src_folder, dst_folder, 'face',
                                                                 id_mapping[stripped_foldername],
                                                                 f"Processed for GWTG {id_mapping[stripped_foldername]}")
                            if result == 1:
                                log_file.write(f"Success: {foldername} processed successfully.\n")
                                if not os.listdir(dst_folder):
                                    os.rmdir(dst_folder)
                                    log_file.write(
                                        f"Warning: {foldername} resulted in an empty folder and was deleted.\n")
                            else:
                                log_file.write(f"Error: {foldername} failed to process. {self.processor.error}\n")
                                if os.path.exists(dst_folder):
                                    shutil.rmtree(dst_folder)
                        else:
                            log_file.write(f"Skipped: {foldername} not found in Excel mapping.\n")
                    except Exception as e:
                        log_file.write(f"Error: {foldername} failed with error: {str(e)}\n")
                        if os.path.exists(dst_folder):
                            shutil.rmtree(dst_folder)
                    progress = int((i + 1) / total_folders * 100)
                    self.progress_signal.emit(progress)
                self.finished_signal.emit("Process Completed.")
            except Exception as e:
                self.finished_signal.emit(f"Error: {str(e)}")
            self.progress_signal.emit(100)


class DicomApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = DicomProcessor()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('DICOM Processor')
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout()

        self.input_dir_label = QLabel('Input Directory:')
        layout.addWidget(self.input_dir_label)

        self.input_dir_edit = QLineEdit(self)
        layout.addWidget(self.input_dir_edit)

        self.input_dir_button = QPushButton('Browse', self)
        self.input_dir_button.clicked.connect(self.browse_input_dir)
        layout.addWidget(self.input_dir_button)

        self.excel_label = QLabel('Excel File:')
        layout.addWidget(self.excel_label)

        self.excel_edit = QLineEdit(self)
        layout.addWidget(self.excel_edit)

        self.excel_button = QPushButton('Browse', self)
        self.excel_button.clicked.connect(self.browse_excel)
        layout.addWidget(self.excel_button)

        self.output_dir_label = QLabel('Output Directory:')
        layout.addWidget(self.output_dir_label)

        self.output_dir_edit = QLineEdit(self)
        layout.addWidget(self.output_dir_edit)

        self.output_dir_button = QPushButton('Browse', self)
        self.output_dir_button.clicked.connect(self.browse_output_dir)
        layout.addWidget(self.output_dir_button)

        self.process_button = QPushButton('Process', self)
        self.process_button.clicked.connect(self.process_dicom)
        layout.addWidget(self.process_button)

        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def browse_input_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if directory:
            self.input_dir_edit.setText(directory)

    def browse_excel(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Excel File", "", "Excel Files (*.xlsx *.xls)")
        if file_path:
            self.excel_edit.setText(file_path)

    def browse_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_edit.setText(directory)

    def process_dicom(self):
        input_dir = self.input_dir_edit.text()
        excel_path = self.excel_edit.text()
        output_dir = self.output_dir_edit.text()

        if not input_dir or not excel_path or not output_dir:
            QMessageBox.critical(self, "Error", "Please select the input directory, Excel file, and output directory.")
            return

        self.progress_bar.setValue(0)

        self.thread = RenameFoldersThread(input_dir, excel_path, output_dir)
        self.thread.progress_signal.connect(self.update_progress)
        self.thread.finished_signal.connect(self.process_finished)
        self.thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def process_finished(self, message):
        QMessageBox.information(self, "Info", message)
        self.progress_bar.setValue(100)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DicomApp()
    window.show()
    sys.exit(app.exec_())