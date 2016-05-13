import os
import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from copy import copy
from cachetools import cached, LRUCache


__all__ = ['HKMNode', 'HKMNearestNeighbor']


class HKMNode:
    """Representation of a node in a hierarchical K-means model"""

    def __init__(self, clustering=None, nn_model=None, original_idx=None):
        self.clustering = clustering
        self.children = None
        self.nn_model = nn_model
        self.original_idx = original_idx


class HKMNearestNeighbor:
    def __init__(self, branching_factor, max_depth,
                 leaf_size, batch_size=1000, verbose=False):
        """A nearest neighbor algorithm based on a hierarchical k-means
        algorithm with `branching_factor` and a maximum depth of `max_depth`"""
        self.branching_factor = branching_factor
        self.max_depth = max_depth
        self.leaf_size = leaf_size
        self.batch_size = batch_size
        self.verbose = verbose
        self.root = None
        self._n_points = 0
        # to be set when a model is loaded from disk
        self.model_dir = None
        self.inverse_idx = None

    def fit(self, data):
        """
        Runs fitting procedure on
        data: ndarray, array or matrix with all data
        """
        if self.verbose:
            print('Creating root node')

        self.root = self._create_node(data, np.arange(data.shape[0]), 1)
        if self.verbose:
            print('Finished')

    def _create_node(self, data, original_idx, current_depth):
        """Create an HKMNode and recursevily ask to create nodes if maximum
        depth has not been reached"""

        # end of recursion condition
        if current_depth >= self.max_depth or data.shape[0] <= self.leaf_size:
            if self.verbose:
                self._n_points += data.shape[0]

            nn_model = NearestNeighbors(n_neighbors=min(data.shape[0], self.leaf_size),
                                        metric='cosine', algorithm='brute').fit(data)
            node = HKMNode(nn_model=nn_model,
                           original_idx=original_idx)
        else:
            # it is possible to create a new branch in the data
            # go through each children and cluster them
            # cluster with mini-batch K-means
            clustering = MiniBatchKMeans(n_clusters=self.branching_factor,
                                         batch_size=self.batch_size)
            # get one element from each partition
            labels = clustering.fit_predict(data)

            node = HKMNode(clustering=clustering)
            node.children = []

            for children_id in range(self.branching_factor):
                idx = np.where(labels == children_id)[0]
                node.children.append(self._create_node(
                    data[idx],
                    original_idx[idx],
                    current_depth + 1)
                )
        return node

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.root is None:
            return "Empty tree"
        return self._display_node(self.root, '')

    def _display_node(self, node, levels=''):
        str0 = ''
        if node.children is None:
            str0 += ' leaf node\n'
        else:
            str0 += ' Node\n'
            for i in range(len(node.children)):
                str0 += levels + ' *' + self._display_node(node.children[i],
                                                           levels=levels + '  ')
        return str0

    def predict(self, x):
        """
        Predict the tree position of point
        :param x: data point
        """
        tree_path = [0] * self.max_depth
        i = 0
        current_node = self.root
        while current_node.children is not None:
            cl = current_node.clustering.predict(x)[0]
            tree_path[i] = cl
            i += 1
            current_node = current_node.children[cl]
        return tuple(tree_path)

    def get_leaf_node(self, x):
        tree_path = [0] * self.max_depth
        i = 0
        current_node = self.root
        while current_node.children is not None:
            cl = current_node.clustering.predict(x)[0]
            tree_path[i] = cl
            i += 1
            current_node = current_node.children[cl]
        return current_node, tree_path

    def _get_nn_model(self, tree_path):
        file_name = os.path.join(self.model_dir, '_'.join(map(str, tree_path)) + '.pickle')
        return joblib.load(file_name)

    def kneighbors(self, x):
        node, tree_path = self.get_leaf_node(x)
        if node.nn_model is None:
            nn_model = self._get_nn_model(tuple(tree_path))
        else:
            nn_model = node.nn_model

        distance, idx = nn_model.kneighbors(x)

        return distance[0], node.original_idx[idx[0]]

    def save_model(self, model_dir):
        """
        Save the model to `model_dir`

        Parameters
        ----------
        model_dir: str, location where model is saved
        """
        if os.path.isdir(model_dir):
            raise Exception('Folder already exists')
        else:
            os.mkdir(model_dir)

        # We clone the instance but do not clone the leaves since we will save them separately
        new_hkmnn_model = HKMNearestNeighbor(self.branching_factor,
                                             self.max_depth,
                                             self.leaf_size,
                                             self.batch_size,
                                             self.verbose)
        new_hkmnn_model.root = self._recursive_save(self.root,
                                                    0,
                                                    [0] * self.max_depth,
                                                    model_dir)
        # save skeleton
        file_name = os.path.join(model_dir, 'skeleton.pickle')
        joblib.dump(new_hkmnn_model, file_name, protocol=2)

    def _recursive_save(self, node, current_depth, tree_path, model_dir):
        """
        Create intermediate nodes of the tree and save leaves
        """
        if node.children is None:
            new_node = HKMNode(original_idx=node.original_idx)
            file_name = os.path.join(model_dir, '_'.join(map(str, tree_path)) + '.pickle')
            joblib.dump(node.nn_model,
                        file_name,
                        protocol=2)
        else:
            new_node = HKMNode(clustering=node.clustering)
            new_node.children = []

            for children_id in range(self.branching_factor):
                new_tree_path = copy(tree_path)
                new_tree_path[current_depth] = children_id
                new_node.children.append(self._recursive_save(
                               node.children[children_id],
                               current_depth+1,
                               new_tree_path,
                               model_dir))

        return new_node

    def _get_idx_paths(self):
        inverse_idx = []
        self.get_idx_node(self.root,
                          0,
                          [0]*self.max_depth,
                          inverse_idx)
        return inverse_idx

    def get_idx_node(self, node, current_depth, tree_path, inverse_idx):
        if node.children is None:
            inverse_idx.append([node.original_idx, tree_path, node])
        else:
            for children_id in range(self.branching_factor):
                new_tree_path = copy(tree_path)
                new_tree_path[current_depth] = children_id
                self.get_idx_node(node.children[children_id],
                                  current_depth+1,
                                  new_tree_path,
                                  inverse_idx)

    @staticmethod
    def load_model(model_dir, leaf_cache_size=10, points_cache_size=100000):
        """
        Loads a model from `index_dir` and returns an instance of `HKMNearestNeighbor`
        :param model_dir location where model is saved
        :param leaf_cache_size how many leaves to keep in the cache
        :param points_cache_size how many individual points to keep in cache
        """
        # load skeleton
        file_name = os.path.join(model_dir, 'skeleton.pickle')
        new_hkmnn_model = joblib.load(file_name)
        new_hkmnn_model.model_dir = model_dir
        # compute inverse index
        new_hkmnn_model.inverse_idx = new_hkmnn_model._get_idx_paths()

        # cache calls to get_vector and _get_nn_model
        get_nn_model_cache = LRUCache(maxsize=leaf_cache_size)
        get_vector_cache = LRUCache(maxsize=points_cache_size)
        new_hkmnn_model.get_vector = cached(get_vector_cache)(new_hkmnn_model.get_vector)
        new_hkmnn_model._get_nn_model = cached(get_nn_model_cache)(new_hkmnn_model._get_nn_model)

        return new_hkmnn_model

    def get_vector(self, doc_id):
        """Get the vectors stored in the leaves
        :param doc_id id of document"""
        if not hasattr(self, 'model_dir'):
            raise Exception("Only works with models in disk")
        tree_path, internal_doc_id, node = next([path,
                                                 next(i for i in range(len(idx)) if idx[i] == doc_id),
                                                 node]
                                                for idx, path, node
                                                in self.inverse_idx
                                                if doc_id in idx)
        if node.nn_model is None:
            nn_model = self._get_nn_model(tuple(tree_path))
        else:
            nn_model = node.nn_model

        return nn_model._fit_X[[internal_doc_id]]

    def clear_index(self):
        self.clear_node(self.root)

    def clear_node(self, node):
        if node.children is None:
            del node.nn_model
        else:
            for children_id in range(self.branching_factor):
                self.clear_node(node.children[children_id])


def get_lastbranch(obj, x):
    tree_path = [0] * obj.max_depth
    current_depth = 0
    current_node = obj.root
    while current_node.children[0].children is not None:
        cl = current_node.clustering.predict(x)[0]
        tree_path[current_depth] = cl
        current_depth += 1
        current_node = current_node.children[cl]
    return current_node, tree_path, current_depth


def kneighbors_expanded(obj, x, n_siblings=3):
    branch_node, tree_path, current_depth = get_lastbranch(obj, x)
    # review siblings in order
    distance_to_centroids = branch_node.clustering.transform(x)[0]
    all_distances = []
    all_original_idx = []
    for cl in np.argsort(distance_to_centroids)[:(n_siblings+1)]:
        tree_path[current_depth] = cl
        node = branch_node.children[cl]
        if node.nn_model is None:
            nn_model = obj._get_nn_model(tuple(tree_path))
        else:
            nn_model = node.nn_model

        distance, idx = nn_model.kneighbors(x)
        all_distances.append(distance[0])
        all_original_idx.append(node.original_idx[idx[0]])

    all_distances = np.hstack(all_distances)
    all_original_idx = np.hstack(all_original_idx)
    sorted_idx = np.argsort(all_distances)
    return all_distances[sorted_idx], all_original_idx[sorted_idx]
