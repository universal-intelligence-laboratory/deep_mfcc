import os
import sys
import tarfile
import urllib

  def maybe_download_and_extract_dataset(data_url, dest_directory):
    """Download and extract data set tar file.

    If the data set we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a
    directory.
    If the data_url is none, don't download anything and expect the data
    directory to contain the correct files already.

    Args:
      data_url: Web location of the tar file containing the data set.
      dest_directory: File path to extract data to.
    """
    if not data_url:
      return
    if not os.path.exists(dest_directory):
      os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

      def _progress(count, block_size, total_size):
        sys.stdout.write(
            '\r>> Downloading %s %.1f%%' %
            (filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

      try:
        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
      except:
        raise
      statinfo = os.stat(filepath)
      tarfile.open(filepath, 'r:gz').extractall(dest_directory)