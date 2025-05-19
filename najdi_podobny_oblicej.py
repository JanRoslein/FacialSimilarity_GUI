import os
import shutil
import pickle
import pandas as pd
from PIL import Image, ImageFile, ImageTk
import numpy as np
from deepface import DeepFace
from retinaface import RetinaFace
import tkinter as tk
from tkinter import filedialog, messagebox
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

def align_faces(image_path):
    try:
        image = Image.open(image_path)
        image_np = np.array(image)
        faces = RetinaFace.extract_faces(img_path=image_path, align=True)
        return [Image.fromarray(face) for face in faces]
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return []

def process_image(aligned_image, slozka_s_obrazky, prah, vystupni_slozka):
    aligned_image_np = np.array(aligned_image)  # Convert PIL Image to numpy array
    dfs = DeepFace.find(
        img_path=aligned_image_np,
        db_path=slozka_s_obrazky,
        model_name="ArcFace",
        distance_metric="cosine",
        detector_backend="retinaface",
        enforce_detection=False
    )
    if len(dfs) > 0 and all(isinstance(df, pd.DataFrame) for df in dfs):
        dfs = pd.concat(dfs, ignore_index=True)
    if not dfs.empty:
        results = []
        for index, row in dfs.iterrows():
            puvodni_nazev_souboru = os.path.basename(row['identity'])
            hodnota_podobnosti = 1 - row['distance']
            if hodnota_podobnosti > prah:
                cesta_k_vystupu = os.path.join(vystupni_slozka, puvodni_nazev_souboru)
                shutil.copy2(row['identity'], cesta_k_vystupu)
                results.append({'cesta_k_obrazku': row['identity'], 'skore': hodnota_podobnosti})
        return results, dfs
    return [], pd.DataFrame()

def najdi_podobne_obliceje(cesta_k_obrazku, slozka_s_obrazky, prah, vystupni_slozka):
    print(f"Složka s obrázky: {slozka_s_obrazky}")
    aligned_faces = align_faces(cesta_k_obrazku)
    if not aligned_faces:
        print("Chyba při zpracování vstupního obrázku.")
        return
    results = []
    for aligned_image in tqdm(aligned_faces, desc="Processing images"):
        result_set, dfs = process_image(aligned_image, slozka_s_obrazky, prah, vystupni_slozka)
        results.append((result_set, dfs))
    global vysledky
    vysledky = [result for result_set, _ in results for result in result_set]
    dfs = pd.concat([dfs for _, dfs in results], ignore_index=True)
    if not dfs.empty:
        print("\nPodobné obličeje:")
        for vysledek in vysledky:
            print(f"{vysledek['cesta_k_obrazku']} - Skóre podobnosti: {vysledek['skore']}")
        exportuj_similarity_matrix(dfs, slozka_s_obrazky)
    else:
        print("Nebyly nalezeny žádné podobné obličeje.")

def exportuj_similarity_matrix(dfs, slozka_s_obrazky):
    similarity_matrix_file = os.path.join(slozka_s_obrazky, "similarity_matrix.pkl")
    with open(similarity_matrix_file, "wb") as file:
        pickle.dump(dfs, file)
    print(f"Similarity matrix exportována do: {similarity_matrix_file}")

def browse_file(entry, label):
    filename = filedialog.askopenfilename()
    entry.delete(0, tk.END)
    entry.insert(0, filename)
    if filename:
        image = Image.open(filename)
        image.thumbnail((100, 100))
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo

def browse_folder(entry):
    foldername = filedialog.askdirectory()
    entry.delete(0, tk.END)
    entry.insert(0, foldername)

def run_script():
    cesta_k_obrazku = entry_cesta_k_obrazku.get()
    slozka_s_obrazky = entry_slozka_s_obrazky.get()
    prah = float(entry_prah.get())
    vystupni_slozka = entry_vystupni_slozka.get()
    
    najdi_podobne_obliceje(cesta_k_obrazku, slozka_s_obrazky, prah, vystupni_slozka)

def create_gui():
    root = tk.Tk()
    root.title("Find Similar Faces")
    
    tk.Label(root, text="Cesta k obrazku:").grid(row=0, column=0, padx=10, pady=10)
    global entry_cesta_k_obrazku
    entry_cesta_k_obrazku = tk.Entry(root, width=50)
    entry_cesta_k_obrazku.grid(row=0, column=1, padx=10, pady=10)
    global label_thumbnail
    label_thumbnail = tk.Label(root)
    label_thumbnail.grid(row=0, column=3, padx=10, pady=10)
    tk.Button(root, text="Browse", command=lambda: browse_file(entry_cesta_k_obrazku, label_thumbnail)).grid(row=0, column=2, padx=10, pady=10)
    
    tk.Label(root, text="Slozka s obrazky:").grid(row=1, column=0, padx=10, pady=10)
    global entry_slozka_s_obrazky
    entry_slozka_s_obrazky = tk.Entry(root, width=50)
    entry_slozka_s_obrazky.grid(row=1, column=1, padx=10, pady=10)
    tk.Button(root, text="Browse", command=lambda: browse_folder(entry_slozka_s_obrazky)).grid(row=1, column=2, padx=10, pady=10)
    
    tk.Label(root, text="Prah podbnosti:").grid(row=2, column=0, padx=10, pady=10)
    global entry_prah
    entry_prah = tk.Entry(root, width=50)
    entry_prah.grid(row=2, column=1, padx=10, pady=10)
    
    tk.Label(root, text="Vystupni slozka:").grid(row=3, column=0, padx=10, pady=10)
    global entry_vystupni_slozka
    entry_vystupni_slozka = tk.Entry(root, width=50)
    entry_vystupni_slozka.grid(row=3, column=1, padx=10, pady=10)
    tk.Button(root, text="Browse", command=lambda: browse_folder(entry_vystupni_slozka)).grid(row=3, column=2, padx=10, pady=10)
    
    tk.Button(root, text="Run", command=run_script).grid(row=4, column=0, columnspan=3, padx=10, pady=10)
    
    root.mainloop()

create_gui()