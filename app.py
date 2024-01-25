from ImageProcessor import ImageProcessor as Processor
from ImageRenderer import ImageRenderer as Renderer

print("FITS 파일 로드 시작")
s_path = 'resources/S.fit'
h_path = 'resources/H.fit'
o_path = 'resources/O.fit'

print("원본 이미지 SHO로 합성 시작")
original_image = Processor.create_sho_image(s_path, h_path, o_path)

print("밝기 조정 시작")
brightness_factor = 200  # 밝기 조정 인자
rgb_brightened = Processor.adjust_brightness(original_image, brightness_factor)

print("대비 조정 시작")
contrast_factor = 1.2   # 대비 조정 인자
rgb_contrasted = Processor.adjust_contrast(rgb_brightened, contrast_factor)

print("색상 균형 조정 시작")
auto_balanced = Processor.match_histograms(rgb_contrasted)

print("asinh 스트레칭 적용 시작")
none_linear_factor = 100  # 스트레칭의 강도를 조절할 인자
asinh_stretched = Processor.asinh_stretch(auto_balanced, none_linear_factor)

print("자동 스트레칭 시작")
auto_strectched = Processor.auto_stretch(auto_balanced)

print("밝기 조정 시작")
rgb_brightened2 = Processor.adjust_brightness(auto_strectched, 0.8)

print("대비 조정 시작")
rgb_contrasted2 = Processor.adjust_contrast(rgb_brightened2, 5)

print("자동 색상 균형 조정 시작")
auto_color_balance = Processor.auto_color_balance(rgb_contrasted2)

# 별 제거
#rgb_starless = Processor.remove_stars(rgb_contrasted2)

print("렌더링 시작")
renderer = Renderer()
renderer.render(auto_color_balance)

print("TIFF 파일로 저장 시작")
renderer.save_as_tif(auto_color_balance, 'result.tif')

print("모든 과정 완료")