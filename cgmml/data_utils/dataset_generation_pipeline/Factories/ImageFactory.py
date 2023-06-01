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
        zip_input_full_path = f"{input_dir}/{artifact_dict['file_path']}"

        pil_im = Image.open(zip_input_full_path)
        pil_im = pil_im.rotate(-90, expand=True)
        rgb_height, rgb_width = pil_im.width, pil_im.height  # Weird switch
        pil_im = pil_im.resize((rgb_height, rgb_width), Image.ANTIALIAS)
        layers = np.asarray(pil_im)

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
