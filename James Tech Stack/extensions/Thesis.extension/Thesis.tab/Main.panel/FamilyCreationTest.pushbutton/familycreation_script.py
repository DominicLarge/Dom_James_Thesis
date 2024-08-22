#! python3
import os
import shutil
import glob

import Autodesk.Revit.DB as DB
import Autodesk.Revit.UI as UI
import Autodesk.Revit.ApplicationServices as AS
import Autodesk.Revit.Creation as CR

doc = __revit__.ActiveUIDocument.Document
uidoc = __revit__.ActiveUIDocument
app = __revit__.Application

library_path = "C:\Dom_James_Thesis\James Tech Stack\Revit Families"

#replace with Dom's file selection
template = "Table - Square.rfa"

new_name_input = "Table1"
new_name = new_name_input +".rfa"


# #Find family and copy, rename, load, and delete it
loaded_family = None
def load_family(family, library, new_name, doc):

    original_path = None
    for root, dir, files in os.walk(library):
        # print(files, dir)
        if family in files:
            original_path = os.path.join(root, family)
            # print(original_path)
            break
        
        else:
            print(f"File {family} not found in {library}")
    
    if original_path:
        new_filepath = os.path.join(os.path.dirname(original_path), new_name)
        # print(new_filepath)
        shutil.copy2(original_path, new_filepath)

        t1 = DB.Transaction(doc, "Load family")
        t1.Start()
        family_id = doc.LoadFamily(new_filepath)
        print(family_id)
        
        t1.Commit()

        os.remove(new_filepath)
 

load_family(template, library_path, new_name, doc)

# uidoc.PromptForFamilyInstancePlacement(family_symbol)