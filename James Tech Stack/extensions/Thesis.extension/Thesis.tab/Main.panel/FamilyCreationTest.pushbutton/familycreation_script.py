#! python3
import os
import shutil
import glob
import clr

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

new_name = "Table1.rfa"

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

        box = clr.StrongBox[Family]()
        t1 = DB.Transaction(doc, "Load family")
        t1.Start()
        success = doc.LoadFamily(new_filepath, box)

    
        if success:
            loaded_family = family.Value
            name = loaded_family.Name
            print(name)
            # Get the first FamilySymbol from the loaded family
            family_symbol_id = loaded_family.GetFamilySymbolIds().GetEnumerator().Current
            family_symbol = doc.GetElement(family_symbol_id)
        
            # Ensure the FamilySymbol is active
            if not family_symbol.IsActive:
                family_symbol.Activate()
                doc.Regenerate()


        
        t1.Commit()

        os.remove(new_filepath)
 

load_family(template, library_path, new_name, doc)

uidoc.PromptForFamilyInstancePlacement(family_symbol)