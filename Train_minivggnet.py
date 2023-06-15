# import các thư viện
from preprocessing import ImageToArrayPreprocessor
from preprocessing import SimplePreprocessor
from preprocessing import SimpleDatasetLoader
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nn.conv import MiniVGGNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import SGD
import os


def trainModel():
    d = "dataset"
    label_names = list(
        filter(lambda x: os.path.isdir(os.path.join(d, x)), os.listdir(d))
    )
    imagePaths = list(paths.list_images("dataset"))
    # Bước 1. Chuẩn bị dữ liệu
    print(imagePaths)
    # Khởi tạo tiền xử lý ảnh
    sp = SimplePreprocessor(32, 32)  # Thiết lập kích thước ảnh 32 x 32
    iap = ImageToArrayPreprocessor()  # Gọi hàm để chuyển ảnh sang mảng

    # Nạp dataset từ đĩa
    print("[INFO] Nạp ảnh...")

    sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
    (data, labels) = sdl.load(imagePaths, verbose=500)
    data = data.astype("float") / 255.0

    # Chia tách dữ liệu vào 02 tập, training: 75% và testing: 25%
    (trainX, testX, trainY, testY) = train_test_split(
        data, labels, test_size=0.25, random_state=42
    )

    # Chuyển dữ liệu nhãn ở số nguyên vào biểu diễn dưới dạng vectors

    trainY = LabelBinarizer().fit_transform(trainY)
    testY = LabelBinarizer().fit_transform(testY)

    # Khởi tạo danh sách các label cho tập dữ liệu flowers

    # Bước 2: Khởi tạo bộ tối ưu và model
    print("[INFO]: Biên dịch model....")
    # Các tham số bộ tối ưu:
    #   - learning_rate: Tốc dộ học
    #   - decay: sử dụng để giảm từ từ tốc độ học theo thời gian
    #            được tính bằng Tốc độ học /tổng epoch. Dùng để tránh overfitting
    #            và tăng độ chính xác khi tranningpy
    #   - momentum: Hệ số quán tính
    #   - nesterov = True: sử dụng phương pháp tối ưu Nestrov accelerated gradient
    optimizer = SGD(learning_rate=0.01, decay=0.01 / 60, momentum=0.9, nesterov=True)
    model = MiniVGGNet.build(width=32, height=32, depth=3, classes=len(label_names))
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    # Bước 3: Train the network
    print("[INFO]: Đang trainning....")
    H = model.fit(
        trainX,
        trainY,
        validation_data=(testX, testY),
        batch_size=64,
        epochs=60,
        verbose=1,
    )

    model.save("miniVGGNet.hdf5")  # Lưu model
    model.summary()  # Hiển thị tóm tắt các tham số của model

    # Bước 4: Đánh giá mạng
    # print("[INFO]: Đánh giá model....")
    # predictions = model.predict(testX, batch_size=64)
    # print(
    #     classification_report(
    #         testY.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names
    #     )
    # )


# Vẽ biểu đồ
# Vẽ kết quả trainning: mất mát (loss) và độ chính xác (accuracy) quá trình trainning
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, 60), H.history["loss"], label="Mất mát khi trainning")
# plt.plot(np.arange(0, 60), H.history["val_loss"], label="Mất mát validation")
# plt.plot(np.arange(0, 60), H.history["accuracy"], label="Độ chính xác khi trainning")
# plt.plot(np.arange(0, 60), H.history["val_accuracy"], label="Độ chính xác validation ")
# plt.title("Biểu đồ hiển thị mất mát và độ chính xác khi Training")
# plt.xlabel("Epoch #")
# plt.ylabel("Mất mát/Độ chính xác")
# plt.legend()
# plt.show()
