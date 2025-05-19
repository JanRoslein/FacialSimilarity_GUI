import os
import shutil
import pickle
import pandas as pd
from PIL import Image, ImageFile
import numpy as np
from deepface import DeepFace
import argparse
from retinaface import RetinaFace
from multiprocessing import Pool

ImageFile.LOAD_TRUNCATED_IMAGES = True

def align_faces(image_path):
    try:
        image = Image.open(image_path)
        faces = RetinaFace.extract_faces(img_path=image_path, align=True)
        aligned_faces = [Image.fromarray(face) for face in faces]
        return aligned_faces
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return []

def process_image(args):
    aligned_image, slozka_s_obrazky, prah, vystupni_slozka = args
    aligned_image_np = np.array(aligned_image)
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
    args = [(aligned_image, slozka_s_obrazky, prah, vystupni_slozka) for aligned_image in aligned_faces]
    with Pool() as pool:
        results = pool.map(process_image, args)
    global vysledky
    vysledky = [result for result_set, _ in results for result in result_set]
    dfs = pd.concat([dfs for _, dfs in results], ignore_index=True)
    if not dfs.empty:
        print("\nPodobné obličeje:")
        for vysledek in vysledky:
            print(f"{vysledek['cesta_k_obrazku']} - Skóre podobnosti: {vysledek['skore']:.3f}")
        exportuj_similarity_matrix(dfs, slozka_s_obrazky)
    else:
        print("Nebyly nalezeny žádné podobné obličeje.")

def exportuj_similarity_matrix(dfs, slozka_s_obrazky):
    similarity_matrix_file = os.path.join(slozka_s_obrazky, "similarity_matrix.pkl")
    with open(similarity_matrix_file, "wb") as file:
        pickle.dump(dfs, file)
    print(f"Similarity matrix exportována do: {similarity_matrix_file}")

def main():
    parser = argparse.ArgumentParser(description="Find similar faces in a folder of images.")
    parser.add_argument("-c", "--cesta_k_obrazku", type=str, help="Cesta k obrazku")
    parser.add_argument("-s", "--slozka_s_obrazky", type=str, help="Cesta ke slozce s obrazky")
    parser.add_argument("-p", "--prah", type=float, help="Prah podbnosti hodnota mezi 0 - 1, pouzij tecku namisto desitinne carky")
    parser.add_argument("-vy", "--vystupni_slozka", type=str, help="Cesta k vystupni slozce")

    args = parser.parse_args()
    najdi_podobne_obliceje(args.cesta_k_obrazku, args.slozka_s_obrazky, args.prah, args.vystupni_slozka)

if __name__ == "__main__":
    main()