from pathlib import Path

from azure.storage.blob import BlobServiceClient


class BlobRepo:
    def download_from_blob_storage(src: str, dest: str, container: str, blob_client: BlobServiceClient) -> None:
        """
        Download a file from Azure Blob Storage and save it to the specified destination.

        Args:
        src (str): The source path of the file in the Blob Storage.
        dest (str): The destination path where the downloaded file will be saved.
        container (str): The name of the container in Blob Storage.
        blob_client (BlobServiceClient): An instance of the Azure BlobServiceClient.

        Returns:
        None
        """
        # Get the BlobClient for the specified container and blob (file)
        blob = blob_client.get_blob_client(container=container, blob=src)

        # Create the destination directory if it doesn't exist
        Path(dest).parent.mkdir(parents=True, exist_ok=True)

        # Download the file from Blob Storage and save it to the destination
        with open(dest, "wb") as download_file:
            content = blob.download_blob().readall()
            download_file.write(content)