
import os
import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
import geojson 
import tifffile
from datetime import datetime
import argparse  # добавляем библиотеку argparse для работы с аргументами командной строки

def load_image(file_path):
    try:
        image = tifffile.imread(file_path)
        
        if image.dtype != np.uint8:
            image = (image / np.max(image) * 255).astype(np.uint8)
        
        if image.shape[2] == 4:
            image = image[:, :, :3]
        
        return image, os.path.basename(file_path)
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None, os.path.basename(file_path)

def process_image(image, georeferenced_image, geotransform, filename):
    if image is None:
        print(f"Skipping image {filename} due to loading error.")
        return None, None

    H = find_homography(image, georeferenced_image)

    if H is not None:
        corner_coords = get_corner_coordinates(H, image.shape[1], image.shape[0], geotransform)
        return corner_coords, H
    else:
        print(f"Failed to find homography for {filename}.")
        return None, None

def find_homography(image1, image2):
    print(f"Processing images with shapes {image1.shape} and {image2.shape}")

    orb = cv2.ORB_create(nfeatures=2000)

    try:
        keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(image2, None)
    except cv2.error as e:
        print(f"Error computing ORB descriptors: {e}")
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    if descriptors1 is not None and descriptors2 is not None:
        matches = bf.match(descriptors1, descriptors2)
    else:
        print("Descriptors are None. Unable to perform matching.")
        return None

    good_matches = sorted(matches, key=lambda x: x.distance)[:80]

    if len(good_matches) < 4:
        print(f"Not enough good matches: {len(good_matches)}")
        return None

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return H

def get_corner_coordinates(H, width, height, geotransform):
    corners = np.float32([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, H)

    corner_coords = []
    for corner in transformed_corners:
        x, y = corner.ravel()
        px, py = rasterio.transform.xy(geotransform, y, x, offset='center')  # Правильный порядок x, y
        corner_coords.append((px, py))

    return corner_coords

def save_geojson(corner_coords, layout_name, crop_name, geotransform, start_time, end_time):
    crs_epsg = 32637
    crs = f"EPSG:{crs_epsg}"

    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "layout_name": layout_name,
                    "crop_name": crop_name,
                    "start": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "end": end_time.strftime("%Y-%m-%dT%H:%M:%S")
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [corner_coords[0][0], corner_coords[0][1]],
                        [corner_coords[1][0], corner_coords[1][1]],
                        [corner_coords[2][0], corner_coords[2][1]],
                        [corner_coords[3][0], corner_coords[3][1]],
                        [corner_coords[0][0], corner_coords[0][1]]
                    ]]
                }
            }
        ]
    }

    geojson_filename = f"{crop_name}_to_{layout_name}_geojson.geojson"
    with open(geojson_filename, 'w') as geojson_file:
        geojson.dump(geojson_data, geojson_file, indent=2)

    print(f"GeoJSON с координатами углов сохранен в файле: {geojson_filename}")


def save_geotiff(image, geotransform, crop_name):
    tiff_filename = f"{crop_name}_aligned.tif"
    num_channels = image.shape[2]

    # Установите точные размеры GeoTIFF файла
    height, width = image.shape[:2]

    # Здесь укажите EPSG код вашей системы координат
    epsg_code = 32637

    # Проверяем, существует ли файл, и удаляем его, если он есть
    if os.path.exists(tiff_filename):
        os.remove(tiff_filename)
        print(f"Существующий файл {tiff_filename} удален перед сохранением нового.")

    with rasterio.open(
        tiff_filename,
        'w',
        driver='GTiff',
        width=width,
        height=height,
        count=num_channels,
        dtype=image.dtype,
        transform=geotransform,
        crs=f"EPSG:{epsg_code}"  # Использование EPSG кода
    ) as dst:
        for i in range(num_channels):
            dst.write(image[:,:,i], i+1)

    print(f"Привязанный снимок сохранен в GeoTIFF файле: {tiff_filename}")

if __name__ == '__main__':
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Process images with georeferencing')
    parser.add_argument('--crop_name', type=str, help='Path to crop image')
    parser.add_argument('--layout_name', type=str, help='Path to layout image')
    args = parser.parse_args()

    # Проверка наличия аргументов
    if args.crop_name is None or args.layout_name is None:
        parser.print_help()
        exit(1)

    satellite_image_path = args.crop_name
    georeferenced_image_path = args.layout_name

    layout_name = os.path.basename(georeferenced_image_path)
    crop_name = os.path.basename(satellite_image_path)
    start_time = datetime.now()

    with rasterio.open(georeferenced_image_path) as src:
        georeferenced_image = src.read().astype(np.uint8)
        geotransform = src.transform

    if georeferenced_image.shape[0] == 4:
        georeferenced_image = georeferenced_image[:3, :, :].transpose(1, 2, 0)

    print(f"Processing file: {satellite_image_path}")
    image, filename = load_image(satellite_image_path)
    corner_coords, H = process_image(image, georeferenced_image, geotransform, filename)

    if corner_coords is not None and H is not None:
        end_time = datetime.now()
        save_geojson(corner_coords, layout_name, crop_name, geotransform, start_time, end_time)
        save_geotiff(image, geotransform, crop_name)
    