#! python3
import os
import shutil
import glob

import Autodesk.Revit.DB as DB
import Autodesk.Revit.UI as UI
import Autodesk.Revit.ApplicationServices as AS
from Autodesk.Revit.Creation import Document

doc = __revit__.ActiveUIDocument.Document
app = __revit__.Application

library_path = "C:\Dom_James_Thesis\James Tech Stack\Revit Families"

#replace with Dom's file selection
template = "Table - Square.rfa"

new_name = "Table1.rfa"

#Search library for family
def search_file(directory, filename):
    # Create a search pattern
    pattern = os.path.join(directory, f"*{filename}*")
    
    # Use glob to find all matching files
    matching_files = glob.glob(pattern)
    
    return matching_files[0] if matching_files else None

#Find family and copy, rename, load, and delete it
def load_family(family, library, new_name, doc):

    original_path = None
    for root, dir, files in os.walk(library):
        print(files, dir)
        if family in files:
            original_path = os.path.join(root, family)
            print(original_path)
            break
        
        else:
            print(f"File {family} not found in {library}")
    
    if original_path:
        new_filepath = os.path.join(os.path.dirname(original_path), new_name)
        print(new_filepath)
        shutil.copy2(original_path, new_filepath)

        t1 = DB.Transaction(doc, "Load family")
        t1.Start()
        doc.LoadFamily(new_filepath)
        t1.Commit()

        os.remove(new_filepath)


load_family(library_path, template, new_name, doc)
