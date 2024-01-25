from Image import Image
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, maximum_filter
from skimage.feature import peak_local_max

class ImageProcessor:

    @staticmethod
    def _load_fits_data(fits_path):
        with fits.open(fits_path) as hdul:
            data = hdul[0].data
        return data.astype(np.float32)

    @staticmethod
    def create_sho_image(s_path, h_path, o_path):
        s_data = ImageProcessor._load_fits_data(s_path)
        h_data = ImageProcessor._load_fits_data(h_path)
        o_data = ImageProcessor._load_fits_data(o_path)

        # S, H, O 이미지 데이터를 하나의 RGB 이미지로 결합
        combined_data = np.stack((s_data, h_data, o_data), axis=-1)
        return Image(combined_data)

    @staticmethod
    def asinh_stretch(rgb_image, non_linear_factor):
        norm_image = rgb_image.data / rgb_image.data.max()
        stretched_image = np.arcsinh(norm_image * non_linear_factor) / np.arcsinh(non_linear_factor)
        return Image(stretched_image)

    @staticmethod
    def auto_color_balance(rgb_image):
        mean_vals = np.mean(rgb_image.data, axis=(0, 1))
        balance_factors = mean_vals.min() / mean_vals
        balanced_image = rgb_image.data * balance_factors
        return Image(balanced_image)

    @staticmethod
    def match_histograms(image, bins=65536):
        """
        16비트 이미지 데이터의 히스토그램 매칭을 수행하는 함수.
        :param image: Image 인스턴스.
        :param bins: 히스토그램 계산에 사용될 빈(bin)의 수.
        :return: 히스토그램이 매칭된 Image 인스턴스.
        """
        # 내부 히스토그램 매칭 함수
        def _match_histogram(channel, reference):
            channel = np.clip(channel, 0, 1)  # 데이터 클리핑
            reference = np.clip(reference, 0, 1)  # 데이터 클리핑

            # 히스토그램 및 누적분포함수(CDF) 계산
            hist, bin_edges = np.histogram(channel, bins=bins, range=(0, 1))
            cdf = np.cumsum(hist).astype(float) / hist.sum()

            ref_hist, ref_bin_edges = np.histogram(reference, bins=bins, range=(0, 1))
            ref_cdf = np.cumsum(ref_hist).astype(float) / ref_hist.sum()

            # 픽셀 값 변환을 위한 보간 수행
            interp_t_values = np.interp(channel.flatten(), bin_edges[:-1], cdf)
            matched_channel = np.interp(interp_t_values, ref_cdf, ref_bin_edges[:-1])
            return matched_channel.reshape(channel.shape)

        # R, G, B 채널 데이터 추출
        R_channel = image.data[:, :, 0]
        G_channel = image.data[:, :, 1]
        B_channel = image.data[:, :, 2]

        # 가장 넓은 데이터 분포를 가진 채널을 기준으로 선택
        integrals = [np.sum(R_channel), np.sum(G_channel), np.sum(B_channel)]
        reference_channel_index = np.argmax(integrals)
        reference_channel_data = image.data[:, :, reference_channel_index]

        # 매칭된 채널 데이터
        matched_channels = [
            _match_histogram(image.data[:, :, i], reference_channel_data) if i != reference_channel_index else image.data[:, :, i]
            for i in range(3)
        ]

        # 매칭된 채널을 쌓아서 최종 이미지 데이터 생성
        matched_image_data = np.stack(matched_channels, axis=2)

        # Image 인스턴스로 반환
        return Image(matched_image_data)

    @staticmethod
    def remove_green_purple(rgb_image, green_removal_factor=0.7, purple_removal_factor=1):
        new_image = rgb_image.data.copy()
        new_image[..., 1] *= green_removal_factor  # 녹색 억제
        new_image[..., 0] *= purple_removal_factor
        new_image[..., 2] *= purple_removal_factor
        return Image(new_image)

    @staticmethod
    def adjust_brightness(image, brightness_factor):
        brightened_image = np.clip(image.data * brightness_factor, 0, 1)
        return Image(brightened_image)

    @staticmethod
    def adjust_contrast(image, contrast_factor):
        mean = np.mean(image.data)
        contrasted_image = np.clip(((image.data - mean) * contrast_factor + mean), 0, 1)
        return Image(contrasted_image)


    @staticmethod
    def auto_stretch(image, stretch_intensity=100.0):
        """
        이미지의 대비를 조정하는 함수로, PixInsight의 STF 기능과 유사하게 작동합니다.

        :param image: Image 인스턴스.
        :param stretch_intensity: 스트레칭 강도를 결정하는 파라미터.
        :return: 대비가 조정된 Image 인스턴스.
        """
        # 새 이미지 데이터 컨테이너를 생성합니다.
        stretched_data = np.zeros_like(image.data)

        # 각 채널에 대해 STF를 적용합니다.
        for i in range(3):  # 각 채널 R, G, B에 대해
            channel_data = image.data[:, :, i]

            # 밝기의 중간값을 계산합니다.
            mid = np.median(channel_data)

            # 스트레칭 강도를 결정하는 스케일링 팩터를 계산합니다.
            stretch_factor = np.arcsinh(mid * stretch_intensity) / mid if mid > 0 else stretch_intensity

            # 스트레칭 적용
            stretched_channel = np.arcsinh(channel_data * stretch_factor) / np.arcsinh(stretch_intensity)

            # 스트레칭된 채널 데이터를 정규화합니다.
            stretched_data[:, :, i] = stretched_channel / np.max(stretched_channel)

        return Image(stretched_data)

    @staticmethod
    def remove_stars(image, threshold=0.8, blur_radius=2):
        """
        별을 제거하고 남은 영역을 부드럽게 채우는 메서드.

        :param image: Image 인스턴스.
        :param threshold: 별로 간주되는 밝기 임계값.
        :param blur_radius: 가우시안 블러의 반경.
        :return: 별이 제거된 Image 인스턴스.
        """
        processed_data = np.copy(image.data)

        # 별로 간주될 수 있는 밝은 영역 검출
        star_mask = np.zeros_like(image.data, dtype=bool)
        for i in range(3):
            channel_data = image.data[:, :, i]
            local_max = peak_local_max(image.data, min_distance=1, threshold_abs=0.8)
            max_filter = maximum_filter(channel_data, size=3)
            star_mask[:, :, i] = (channel_data == max_filter) & local_max

        # 별 영역만 블러 처리
        for i in range(3):
            channel_data = processed_data[:, :, i]
            blurred_channel = gaussian_filter(channel_data, sigma=blur_radius)
            channel_data[star_mask[:, :, i]] = blurred_channel[star_mask[:, :, i]]

        return Image(processed_data)

