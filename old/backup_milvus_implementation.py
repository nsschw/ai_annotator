#import Milvi
#
#class MilvusDB:
#    """
#    Based on MilvusLite, therefore not as fuctional
#    Some functions might be weird, bc MilvusLite has a reduced functioality -> Upgrade if needed...
#
#    """
#
#    def __init__(self, path: str, **kwargs) -> None:        
#        self.client = MilvusClient(path)
#        self.collection_name : str = kwargs.get("collection_name", "Demo")
#
#        if self.client.has_collection(collection_name=self.collection_name):
#            pass
#        else: 
#            self.client.create_collection(
#                collection_name=self.collection_name,
#                dimension=768,
#            )
#
#
#    def insert_data(self, data: list[dict]) -> None:
#        # check if consistent with available data
#
#        # insert data
#        self.client.insert(
#            collection_name=self.collection_name,
#            data=data)
#         
#
#    def search(self, vector: np.array, k: int=3):
#
#        res = self.client.search(
#            collection_name="demo_collection",
#            data=[vector],
#            limit=k,
#            output_fields=["text", "subject"],
#            )