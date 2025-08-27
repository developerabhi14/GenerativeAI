from langchain_community.document_loaders import CSVLoader

loader=CSVLoader(file_path="Social_Network_Ads.csv")

data=loader.load()

# one document object for each row

print(data[0])