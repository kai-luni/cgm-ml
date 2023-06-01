from pathlib import Path
import pickle
from PIL import Image
import numpy as np

class ImageFactory:
    def process_rgb(self, artifact_dict, input_dir : str, output_dir : str) -> str:
        """
        Process RGB data, save it as a pickle file, and return the file path.

        Args:
        artifact_dict (dict): Dictionary containing artifact information.
        input_dir (str): Input directory containing the RGB files.
        output_dir (str): Output directory where the processed pickle files will be saved.

        Returns:
        str: File path of the saved pickle file.
        """
        RGB_HEIGHT = 1440
        RGB_WEIGHT = 1080
        zip_input_full_path = f"{input_dir}/{artifact_dict['file_path']}"
        layers = self.read_rgb_data(zip_input_full_path, RGB_HEIGHT, RGB_WEIGHT)
        timestamp = artifact_dict['timestamp']
        scan_id = artifact_dict['scan_id']
        scan_step = artifact_dict['scan_step']
        order_number = artifact_dict['order_number']
        person_id = artifact_dict['person_id']
        pickle_output_path = f"scans/{person_id}/{scan_step}/pc_{scan_id}_{timestamp}_{scan_step}_{order_number}.p"
        target_dict = {**artifact_dict}
        
        # Write into pickle
        pickle_output_full_path = f"{output_dir}/{pickle_output_path}"
        Path(pickle_output_full_path).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump((layers, target_dict), open(pickle_output_full_path, "wb"))

        return pickle_output_full_path

    def read_rgb_data(self, rgb_fpath : str):
        """
        Process RGB dataset by loading, rotating, and resizing the image.

        Args:
        rgb_fpath (str): File path of the RGB image.

        Returns:
        numpy.ndarray: Resized and rotated RGB image as a NumPy array, or None if the file path is not provided.
        """
        if rgb_fpath:
            pil_im = Image.open(rgb_fpath)
            pil_im = pil_im.rotate(-90, expand=True)
            rgb_height, rgb_width = pil_im.width, pil_im.height  # Weird switch
            pil_im = pil_im.resize((rgb_height, rgb_width), Image.ANTIALIAS)
            rgb_array = np.asarray(pil_im)
        else:
            rgb_array = None

        return rgb_array