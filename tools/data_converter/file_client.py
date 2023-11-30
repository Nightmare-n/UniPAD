import cv2
import numpy as np
from os.path import splitext
from typing import Any, Generator, Iterator, Optional, Tuple, Union
from pathlib import Path
import pdb
import os
import tensorflow as tf
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
import tempfile
import pickle
import io
import os
import json


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = os.path.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


class BaseStorageBackend(metaclass=ABCMeta):
    """Abstract class of storage backends.

    All backends need to implement two apis: ``get()`` and ``get_text()``.
    ``get()`` reads the file as a byte stream and ``get_text()`` reads the file
    as texts.
    """

    # a flag to indicate whether the backend can create a symlink for a file
    _allow_symlink = False

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def allow_symlink(self):
        return self._allow_symlink

    @abstractmethod
    def get(self, filepath):
        pass

    @abstractmethod
    def get_text(self, filepath):
        pass


class PetrelBackend(BaseStorageBackend):
    def __init__(self,
                 path_mapping: Optional[dict] = None,
                 enable_mc: bool = True, **kwargs):
        try:
            from petrel_client import client
        except ImportError:
            raise ImportError('Please install petrel_client to enable '
                              'PetrelBackend.')

        self._client = client.Client(enable_mc=enable_mc)
        assert isinstance(path_mapping, dict) or path_mapping is None
        self.path_mapping = path_mapping

    def _map_path(self, filepath: Union[str, Path]) -> str:
        """Map ``filepath`` to a string path whose prefix will be replaced by
        :attr:`self.path_mapping`.

        Args:
            filepath (str): Path to be mapped.
        """
        filepath = str(filepath)
        if self.path_mapping is not None:
            for k, v in self.path_mapping.items():
                filepath = filepath.replace(k, v)
        return filepath

    def get(self, filepath: Union[str, Path], update_cache: bool = False) -> memoryview:
        """Read data from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            memoryview: A memory view of expected bytes object to avoid
                copying. The memoryview object can be converted to bytes by
                ``value_buf.tobytes()``.
        """
        filepath = self._map_path(filepath)
        value = self._client.Get(filepath, update_cache=update_cache)
        value_buf = memoryview(value)
        return value_buf

    def get_text(self,
                 filepath: Union[str, Path],
                 encoding: str = 'utf-8',
                 update_cache: bool = False) -> str:
        """Read data from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.
        """
        return str(self.get(filepath, update_cache), encoding=encoding)

    def put(self, obj: bytes, filepath: Union[str, Path], update_cache: bool = False) -> None:
        """Save data to a given ``filepath``.

        Args:
            obj (bytes): Data to be saved.
            filepath (str or Path): Path to write data.
        """
        filepath = self._map_path(filepath)
        self._client.put(filepath, obj)
        if update_cache:
            self.get(filepath, update_cache)

    def put_text(self,
                 obj: str,
                 filepath: Union[str, Path],
                 encoding: str = 'utf-8',
                 update_cache: bool = False) -> None:
        """Save data to a given ``filepath``.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str): The encoding format used to encode the ``obj``.
                Default: 'utf-8'.
        """
        self.put(bytes(obj, encoding=encoding), filepath, update_cache)

    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path exists.

        Args:
            filepath (str or Path): Path to be checked whether exists.

        Returns:
            bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.
        """
        filepath = self._map_path(filepath)
        return self._client.contains(filepath) or self._client.isdir(filepath)

    def isdir(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a directory.

        Args:
            filepath (str or Path): Path to be checked whether it is a
                directory.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a directory,
            ``False`` otherwise.
        """
        filepath = self._map_path(filepath)
        return self._client.isdir(filepath)

    def isfile(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a file.

        Args:
            filepath (str or Path): Path to be checked whether it is a file.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a file, ``False``
            otherwise.
        """
        filepath = self._map_path(filepath)
        return self._client.contains(filepath)
    
    @contextmanager
    def get_local_path(
            self,
            filepath: Union[str, Path],
            update_cache: bool = False) -> Generator[Union[str, Path], None, None]:
        """Download a file from ``filepath`` and return a temporary path.

        ``get_local_path`` is decorated by :meth:`contxtlib.contextmanager`. It
        can be called with ``with`` statement, and when exists from the
        ``with`` statement, the temporary path will be released.

        Args:
            filepath (str | Path): Download a file from ``filepath``.

        Examples:
            >>> client = PetrelBackend()
            >>> # After existing from the ``with`` clause,
            >>> # the path will be removed
            >>> with client.get_local_path('s3://path/of/your/file') as path:
            ...     # do something here

        Yields:
            Iterable[str]: Only yield one temporary path.
        """
        filepath = self._map_path(filepath)
        assert self.isfile(filepath)
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(self.get(filepath, update_cache))
            f.close()
            yield f.name
        finally:
            os.remove(f.name)

    def list_dir_or_file(self,
                         dir_path: Union[str, Path],
                         list_dir: bool = True,
                         list_file: bool = True,
                         suffix: Optional[Union[str, Tuple[str]]] = None,
                         recursive: bool = False) -> Iterator[str]:
        """Scan a directory to find the interested directories or files in
        arbitrary order.
        Note:
            Petrel has no concept of directories but it simulates the directory
            hierarchy in the filesystem through public prefixes. In addition,
            if the returned path ends with '/', it means the path is a public
            prefix which is a logical directory.
        Note:
            :meth:`list_dir_or_file` returns the path relative to ``dir_path``.
            In addition, the returned path of directory will not contains the
            suffix '/' which is consistent with other backends.
        Args:
            dir_path (str | Path): Path of the directory.
            list_dir (bool): List the directories. Default: True.
            list_file (bool): List the path of files. Default: True.
            suffix (str or tuple[str], optional):  File suffix
                that we are interested in. Default: None.
            recursive (bool): If set to True, recursively scan the
                directory. Default: False.
        Yields:
            Iterable[str]: A relative path to ``dir_path``.
        """
        dir_path = self._map_path(dir_path)
        if list_dir and suffix is not None:
            raise TypeError(
                '`list_dir` should be False when `suffix` is not None')

        if (suffix is not None) and not isinstance(suffix, (str, tuple)):
            raise TypeError('`suffix` must be a string or tuple of strings')

        # Petrel's simulated directory hierarchy assumes that directory paths
        # should end with `/`
        if not dir_path.endswith('/'):
            dir_path += '/'

        root = dir_path

        def _list_dir_or_file(dir_path, list_dir, list_file, suffix,
                              recursive):
            for path in self._client.list(dir_path):
                # the `self.isdir` is not used here to determine whether path
                # is a directory, because `self.isdir` relies on
                # `self._client.list`
                if path.endswith('/'):  # a directory path
                    next_dir_path = os.path.join(dir_path, path)
                    if list_dir:
                        # get the relative path and exclude the last
                        # character '/'
                        rel_dir = next_dir_path[len(root):-1]
                        yield rel_dir
                    if recursive:
                        yield from _list_dir_or_file(next_dir_path, list_dir,
                                                     list_file, suffix,
                                                     recursive)
                else:  # a file path
                    absolute_path = os.path.join(dir_path, path)
                    rel_path = absolute_path[len(root):]
                    if (suffix is None
                            or rel_path.endswith(suffix)) and list_file:
                        yield rel_path

        return _list_dir_or_file(dir_path, list_dir, list_file, suffix,
                                 recursive)

    def load_pickle(self, filepath, update_cache: bool = False):
        return pickle.load(io.BytesIO(self.get(filepath, update_cache)))

    def dump_pickle(self, data, filepath, update_cache: bool = False):
        with io.BytesIO() as f:
            pickle.dump(data, f)
            self.put(f.getvalue(), filepath, update_cache)

    def save_npy(self, data, filepath, update_cache: bool = False):
        with io.BytesIO() as f:
            np.save(f, data)
            self.put(f.getvalue(), filepath, update_cache)

    def load_npy(self, filepath, update_cache: bool = False):
        return np.load(io.BytesIO(self.get(filepath, update_cache)))

    def load_npy_txt(self, filepath, update_cache: bool = False):
        return np.loadtxt(io.BytesIO(self.get(filepath, update_cache)))

    def load_to_numpy(self, filepath, dtype, update_cache: bool = False):
        return np.frombuffer(self.get(filepath, update_cache), dtype=dtype).copy()

    def load_img(self, filepath, update_cache: bool = False):
        return cv2.imdecode(self.load_to_numpy(filepath, np.uint8, update_cache), cv2.IMREAD_COLOR)

    def load_json(self, filepath, update_cache: bool = False):
        # return json.load(io.BytesIO(self.get(filepath, update_cache)))
        return json.loads(self.get(filepath, update_cache).tobytes())
    
    def dump_json(self, data, filepath, update_cache: bool = False):
        self.put(json.dumps(data).encode('utf-8'), filepath, update_cache)
        
    def readlines(self, filepath, update_cache: bool = False):
        # return self.get_text(filepath, update_cache=update_cache).splitlines()
        with io.BytesIO(self.get(filepath, update_cache)) as f:
            lines = f.readlines()
        lines = list(map(lambda x: str(x, encoding='utf-8'), lines))
        return lines


class HardDiskBackend(BaseStorageBackend):
    """Raw hard disks storage backend."""

    _allow_symlink = True

    def __init__(self, **kwargs):
        pass

    def get(self, filepath: Union[str, Path], update_cache: bool = False) -> bytes:
        """Read data from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes: Expected bytes object.
        """
        with open(filepath, 'rb') as f:
            value_buf = f.read()
        return value_buf

    def get_text(self,
                 filepath: Union[str, Path],
                 encoding: str = 'utf-8',
                 update_cache: bool = False) -> str:
        """Read data from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.
        """
        with open(filepath, encoding=encoding) as f:
            value_buf = f.read()
        return value_buf

    def put(self, obj: bytes, filepath: Union[str, Path], update_cache: bool = False) -> None:
        """Write data to a given ``filepath`` with 'wb' mode.

        Note:
            ``put`` will create a directory if the directory of ``filepath``
            does not exist.

        Args:
            obj (bytes): Data to be written.
            filepath (str or Path): Path to write data.
        """
        mkdir_or_exist(os.path.dirname(filepath))
        with open(filepath, 'wb') as f:
            f.write(obj)

    def put_text(self,
                 obj: str,
                 filepath: Union[str, Path],
                 encoding: str = 'utf-8',
                 update_cache: bool = False) -> None:
        """Write data to a given ``filepath`` with 'w' mode.

        Note:
            ``put_text`` will create a directory if the directory of
            ``filepath`` does not exist.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.
        """
        mkdir_or_exist(os.path.dirname(filepath))
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(obj)

    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path exists.

        Args:
            filepath (str or Path): Path to be checked whether exists.

        Returns:
            bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.
        """
        return os.path.exists(filepath)

    def isdir(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a directory.

        Args:
            filepath (str or Path): Path to be checked whether it is a
                directory.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a directory,
            ``False`` otherwise.
        """
        return os.path.isdir(filepath)

    def isfile(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a file.

        Args:
            filepath (str or Path): Path to be checked whether it is a file.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a file, ``False``
            otherwise.
        """
        return os.path.isfile(filepath)

    @contextmanager
    def get_local_path(
            self,
            filepath: Union[str, Path],
            update_cache: bool = False) -> Generator[Union[str, Path], None, None]:
        """Only for unified API and do nothing."""
        yield filepath

    def list_dir_or_file(self,
                         dir_path: Union[str, Path],
                         list_dir: bool = True,
                         list_file: bool = True,
                         suffix: Optional[Union[str, Tuple[str]]] = None,
                         recursive: bool = False) -> Iterator[str]:
        """Scan a directory to find the interested directories or files in
        arbitrary order.
        Note:
            :meth:`list_dir_or_file` returns the path relative to ``dir_path``.
        Args:
            dir_path (str | Path): Path of the directory.
            list_dir (bool): List the directories. Default: True.
            list_file (bool): List the path of files. Default: True.
            suffix (str or tuple[str], optional):  File suffix
                that we are interested in. Default: None.
            recursive (bool): If set to True, recursively scan the
                directory. Default: False.
        Yields:
            Iterable[str]: A relative path to ``dir_path``.
        """
        if list_dir and suffix is not None:
            raise TypeError('`suffix` should be None when `list_dir` is True')

        if (suffix is not None) and not isinstance(suffix, (str, tuple)):
            raise TypeError('`suffix` must be a string or tuple of strings')

        root = dir_path

        def _list_dir_or_file(dir_path, list_dir, list_file, suffix,
                              recursive):
            for entry in os.scandir(dir_path):
                if not entry.name.startswith('.') and entry.is_file():
                    rel_path = os.path.relpath(entry.path, root)
                    if (suffix is None
                            or rel_path.endswith(suffix)) and list_file:
                        yield rel_path
                elif os.path.isdir(entry.path):
                    if list_dir:
                        rel_dir = os.path.relpath(entry.path, root)
                        yield rel_dir
                    if recursive:
                        yield from _list_dir_or_file(entry.path, list_dir,
                                                     list_file, suffix,
                                                     recursive)

        return _list_dir_or_file(dir_path, list_dir, list_file, suffix,
                                 recursive)

    def load_pickle(self, filepath, update_cache: bool = False):
        return pickle.load(open(filepath, 'rb'))

    def dump_pickle(self, data, filepath, update_cache: bool = False):
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def save_npy(self, data, filepath, update_cache: bool = False):
        np.save(filepath, data)

    def load_npy(self, filepath, update_cache: bool = False):
        return np.load(filepath)

    def load_npy_txt(self, filepath, update_cache: bool = False):
        return np.loadtxt(filepath)
    
    def load_to_numpy(self, filepath, dtype, update_cache: bool = False):
        return np.fromfile(filepath, dtype=dtype)

    def load_img(self, filepath, update_cache: bool = False):
        return cv2.imread(filepath, cv2.IMREAD_COLOR)

    def load_json(self, filepath, update_cache: bool = False):
        return json.load(open(filepath, 'r'))
        
    def readlines(self, filepath, update_cache: bool = False):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        return lines


from easydict import EasyDict
BACKEND = EasyDict({
    'NAME': 'PetrelBackend',
    'KWARGS': {
        'path_mapping': {
            './data/waymo/': 's3://openmmlab/datasets/detection3d/waymo/',
            'data/waymo/': 's3://openmmlab/datasets/detection3d/waymo/'
        }
    }
})
client = globals()[BACKEND.NAME](
    **BACKEND.get('KWARGS', {})
)

