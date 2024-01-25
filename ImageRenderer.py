import matplotlib.pyplot as plt
import numpy as np
import tifffile

class ImageRenderer:
    def __init__(self):
        pass

    def display_image(self, image, ax):
        if not isinstance(image.data, np.ndarray):
            raise TypeError("Image data must be a numpy array")
        ax.imshow(image.data, origin='upper')
        ax.set_title("Processed Image", fontsize=24)
        ax.axis('off')

    def display_histogram(self, image, ax):
        if not isinstance(image.data, np.ndarray):
            raise TypeError("Image data must be a numpy array")
        if len(image.data.shape) == 3:  # 컬러 이미지
            # 순서를 B, G, R로 변경하여 겹치는 부분이 잘 보이도록 함
            colors = ['blue', 'green', 'red']
            # 알파 값을 더 낮추어 겹치는 부분의 가독성을 향상
            alpha_values = [0.6, 0.7, 0.8]
            for i, color in enumerate(colors):
                ax.hist(image.data[:, :, i].flatten(), bins=1024, color=color, alpha=alpha_values[i])
            ax.legend(["Blue", "Green", "Red"], loc='upper right', fontsize=20)
        else:  # 그레이스케일 이미지
            ax.hist(image.data.flatten(), bins=2048, color='black', alpha=0.5)

        # X축과 Y축 레이블 추가
        ax.set_xlabel('Intensity Value', fontsize=20)
        ax.set_ylabel('Pixel Count', fontsize=20)

        # 그래프 타이틀 설정
        ax.set_title('Image Histogram', fontsize=24)

        # 변경사항 적용
        ax.figure.canvas.draw()



    def render(self, image):
        # 해상도 설정 (dpi: dots per inch)
        plt.figure(dpi=100)  # 더 높은 값으로 해상도를 높일 수 있습니다.

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 24))
        self.display_image(image, ax1)
        self.display_histogram(image, ax2)
        plt.show()

    def save_as_tif(self, image, file_path):
        # Image 인스턴스에서 이미지 데이터를 추출
        image_data = image.data

        # tifffile 모듈의 imwrite 함수를 사용하여 TIFF 파일로 저장
        tifffile.imwrite(file_path, image_data)