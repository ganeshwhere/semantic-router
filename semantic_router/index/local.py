from typing import List, Tuple
from semantic_router.linear import similarity_matrix, top_scores

class LocalIndex:
    def __init__(self, index: np.ndarray = None, routes: List[str] = None, utterances: List[str] = None):
        """
        Initialize the LocalIndex class.

        Args:
            index (np.ndarray, optional): The index array. Defaults to None.
            routes (List[str], optional): The routes list. Defaults to None.
            utterances (List[str], optional): The utterances list. Defaults to None.
        """
        self.index = index
        self.routes = routes
        self.utterances = utterances

    def add(self, route: str, utterance: str, vector: np.ndarray):
        """
        Add a new route, utterance, and vector to the index.

        Args:
            route (str): The route to add.
            utterance (str): The utterance to add.
            vector (np.ndarray): The vector to add.
        """
        if self.index is None:
            self.index = vector.reshape(1, -1)
            self.routes = [route]
            self.utterances = [utterance]
        else:
            self.index = np.vstack([self.index, vector])
            self.routes.append(route)
            self.utterances.append(utterance)

    def get_routes(self) -> List[Tuple[str, str]]:
        """
        Get the routes and utterances stored in the index.

        Returns:
            List[Tuple[str, str]]: A list of tuples containing the route and utterance.
        """
        return list(zip(self.routes, self.utterances))

    def describe(self) -> dict:
        """
        Describe the index.

        Returns:
            dict: A dictionary containing the index type, dimensions, and number of vectors.
        """
        return {
            "type": "local",
            "dimensions": self.index.shape[1] if self.index is not None else 0,
            "vectors": self.index.shape[0] if self.index is not None else 0,
        }

    def query(self, query_vector: np.ndarray, top_k: int = 5) -> Tuple[List[float], List[str]]:
        """
        Query the index with a vector and return the top k routes and their similarity scores.

        Args:
            query_vector (np.ndarray): The vector to query the index with.
            top_k (int, optional): The number of top routes to return. Defaults to 5.

        Returns:
            Tuple[List[float], List[str]]: A tuple containing the top k similarity scores and routes.
        """
        if self.index is None:
            raise ValueError("Index is not populated.")
        sim_matrix = similarity_matrix(query_vector, self.index)
        scores, indices = top_scores(sim_matrix, top_k)
        routes = [self.routes[i] for i in indices]
        return scores, routes

    def delete(self, route_name: str):
        """
        Delete a route and its associated utterance from the index.

        Args:
            route_name (str): The name of the route to delete.
        """
        if self.routes is None:
            raise ValueError("Routes are not populated.")
        delete_idx = [i for i, route in enumerate(self.routes) if route == route_name]
        self.index = np.delete(self.index, delete_idx, axis=0)
        self.routes = [route for route in self.routes if route != route_name]
        self.utterances = [utterance for utterance in self.utterances if self.routes[i] != route_name]
