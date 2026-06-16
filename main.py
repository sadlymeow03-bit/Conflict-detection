from flask import Flask, render_template, request
from ultralytics import YOLO 
import os
import shutil

# Создание экземпляра веб-приложения Flask.
# Flask используется в качестве серверной части для взаимодействия пользователя с моделью компьютерного зрения через браузер.
app = Flask(__name__)

# Создание каталога для временного хранения загруженных пользователем видеозаписей при первом запуске приложения.
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Загрузка обученной модели обнаружения конфликтов.
model = YOLO("runs/detect/train/weights/best.pt")

# Главная страница приложения. Отображает интерфейс загрузки видеофайла для последующего анализа.
@app.route("/")
def index():
    
    return render_template("index.html")

# Обработка загруженного пользователем видео.
@app.route("/predict", methods=["POST"])
def predict():

# Проверка наличия файла в HTTP-запросе.
    if "video" not in request.files:
        return "Файл не выбран"

    file = request.files["video"]

    if file.filename == "":
        return "Файл не выбран"
    
# Сохранение исходного видео на сервере.
    video_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(video_path)

# Удаление результатов предыдущего запуска.
    if os.path.exists("runs/detect/predict"):
        shutil.rmtree("runs/detect/predict")

    conflict_found = False
    conflict_count = 0


# Запуск модели YOLO.
# conf=0.5 означает, что учитываются только те обнаружения, для которых вероятность принадлежности к классу "conflict" составляет не менее 50%.
# save=True сохраняет обработанное видео с нанесёнными ограничивающими рамками вокруг обнаруженных объектов.
    results = model(
        video_path,
        conf=0.5,
        save=True
    )

    print(type(results))
    print(len(results))
   
# Поиск сохранённого моделью видео.
    processed_video = None

    for file in os.listdir("runs/detect/predict"):
        if file.endswith((".mp4", ".avi", ".mov", ".mkv")):
            processed_video = file
            break

    src = os.path.join(
        "runs/detect/predict",
        processed_video
    )

# Анализ результатов детектирования.
# Каждый объект result соответствует отдельному кадру видео. Если на кадре присутствует хотя бы одна ограничивающая рамка, кадр считается содержащим конфликтную ситуацию.
    for result in results:
        if len(result.boxes) > 0:
# Увеличиваем количество кадров, на которых обнаружен конфликт.
            conflict_count += 1
# Фиксируем факт наличия конфликта на видео.
            conflict_found = True

# Передача результатов анализа в HTML.
    shutil.copy(src, "static/result.avi")
    return render_template(
        "result.html",
        result=conflict_found,
        count=conflict_count,
        video="static/result.mp4"
    )
 

if __name__ == "__main__":
    app.run(debug=True)