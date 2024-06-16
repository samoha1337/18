import rasterio
import numpy as np
import cv2
import argparse
def read_image(file_path):
    with rasterio.open(file_path) as src:
        image = src.read([1, 2, 3])
        image = np.stack(image, axis=2)
    return image

def find_and_correct_bad_pixels(image, threshold=355):
    corrected_image = image.copy()
    h, w, _ = image.shape
    
    # Создаем медианный фильтр для всего изображения
    median_filtered = cv2.medianBlur(image, 3)
    
    bad_pixel_report = []
    
    for y in range(1, h-1):
        for x in range(1, w-1):
            pixel = image[y, x]
            neighbors = [
                image[y-1, x], image[y+1, x], image[y, x-1], image[y, x+1],
                image[y-1, x-1], image[y-1, x+1], image[y+1, x-1], image[y+1, x+1]
            ]
            neighbors_mean = np.mean(neighbors, axis=0)
            if np.linalg.norm(pixel - neighbors_mean) > threshold:
                # Заменяем "битый" пиксель на значение из медианного фильтра
                corrected_image[y, x] = median_filtered[y, x]
                # Добавляем в отчет информацию о битом пикселе
                bad_pixel_report.append((y, x, np.argmax(pixel) + 1, pixel[np.argmax(pixel)], median_filtered[y, x][np.argmax(pixel)]))
    
    return corrected_image, bad_pixel_report

def save_image(image, input_file, output_file):
    with rasterio.open(input_file) as src:
        profile = src.profile
        profile.update(
            dtype=rasterio.uint16,
            count=3,
            compress='lzw',  # Использование сжатия LZW для сохранения места
            tiled=True,
            predictor=2  # Используемый предиктор для сжатия TIFF
        )
        with rasterio.open(output_file, 'w', **profile) as dst:
            for i in range(3):
                dst.write(image[:, :, i].astype(rasterio.uint16), i + 1)

def create_report(report_path, bad_pixel_report):
    with open(report_path, "w") as file:
        file.write("Отчет об исправлениях в тестовом формате:\n")
        file.write("[номер строки]; [номер столбца]; [номер канала]; [«битое» значение]; [исправленное значение]\n")
        for idx, (y, x, channel, bad_value, corrected_value) in enumerate(bad_pixel_report):
            file.write(f"{y}; {x}; {channel}; {bad_value}; {corrected_value}\n")


def main(input_file, output_file, report_file):
    image = read_image(input_file)
    corrected_image, bad_pixel_report = find_and_correct_bad_pixels(image)
    save_image(corrected_image, input_file, output_file)
    create_report(report_file, bad_pixel_report)
    print(f"Изображение успешно обработано и сохранено в {output_file}")
    print(f"Создан отчет: {report_file}")

if __name__ == "__main__":
    # input_file = "C:/Users/Костя/Desktop/HOHOTON/source/1_20/crop_2_0_0000.tif"
    # output_file = "C:/Users/Костя/Desktop/HOHOTON/restored.tif"
    # report_file = "C:/Users/Костя/Desktop/HOHOTON/report.txt"
    # main(input_file, output_file, report_file)
    parser = argparse.ArgumentParser(description='Исправление битых пикселей на изображении')
    parser.add_argument('input_file', type=str, help='Путь к входному файлу изображения')
    parser.add_argument('output_file', type=str, help='Путь для сохранения исправленного изображения')
    parser.add_argument('report_file', type=str, help='Путь для сохранения отчета о битых пикселях')
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.report_file)