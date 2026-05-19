from enum import Enum
from pathlib import Path
from typing import List , Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

def load_pdf(path : Path , load_and_split : bool , text_splitter : str) -> List[Tuple[str , str]]:
    '''
    to load the pdf.
    '''
    if not valid_path(path):
        raise ValueError("Enter the Valid Path of pdf. and try again")

    loader = PyPDFLoader(path)

    documents = loader.load_and_split(text_splitter=text_splitter) if (load_and_split == True ) else loader.load() 
    if len(documents) == 0:
        raise ValueError("Give the appropriate pdf with contents")
    results = []

    for document in documents:
        results.append((document.metadata , document.page_content))

    return results
def lazy_load_pdf( path :Path) -> List[Tuple[str , str]]:
    '''
    lazy load the pdf.
    '''
    if not valid_path(path):
        raise ValueError("Enter the Valid Path of pdf. and try again")
    loader = PyPDFLoader(path)
    results = []
    documents = loader.lazy_load()
    if len(documents) == 0 :
        raise ValueError("Give the appropriate pdf with contents")
    for document in documents:
        results.append((document.metadata , document.page_content))

    return results

def load_csv(path : Path , load_and_split: bool , text_splitter : str ,source_column :str ,encoding: str) -> List[Tuple[str , str]]:
    if not valid_path(path):
        raise ValueError('Enter the Valid Path of pdf. and try again')
    loader = CSVLoader(path, source_column=source_column,encoding=encoding)

    documents = loader.load_and_split(text_splitter=text_splitter) if load_and_split else loader.load()
    results = []

    for document in documents:
        results.append((document.metadata , document.page_content))
    return results

def lazy_load_csv(path : Path , source_column : str , meta_columns : str , encoding : str = "") -> List[Tuple[str , str]]:
    if not valid_path(path):
        raise ValueError("Enter the Valid Path of pdf and try again.")
    loader = CSVLoader(path , source_column= source_column, metadata_columns= meta_columns, encoding=encoding)
    results = []
    documents = loader.lazy_load()
    if len(documents) == 0 :
        raise ValueError("Give the appropriate pdf with contents")
    for document in documents:
        results.append((document.metadata , document.page_content))
    return results
def valid_path(path : Path) -> bool:
    if (path.exists()):
        return True
    return False


class DocumentType(Enum):
    PDF = "pdf"
    CSV = "csv"

def load_documents (document_type : str ,path , load_and_split : bool ,
                    text_splitter : str  , lazy_load : bool  = True ,
                    source_column : str = None , meta_columns :str= None,
                    encoding : str = None):
    '''
    wrapper from csv , pdf for direct use with inbuilt parameters 
    '''
    document_type = document_type.lower()
    results = []
    if (document_type == DocumentType.PDF):
        if lazy_load == True:
            results = lazy_load_pdf(path)
        else:
            results = load_pdf(path 
                            , load_and_split 
                            ,text_splitter)
    elif (document_type == DocumentType.CSV):
        if (lazy_load == True):
            results = lazy_load_csv(path 
            , source_column=source_column
            , meta_columns=meta_columns 
            , encoding=encoding)
        else :
            results = load_csv(path 
            , load_and_split=load_and_split
            , text_splitter=text_splitter 
            , source_column=source_column
            , encoding=encoding)
    return results
