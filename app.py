from ImageProcessor import ImageProcessor as Processor
from ImageRenderer import ImageRenderer as Renderer

# FITS 파일 로드
s_path = 'resources/S.fit'
h_path = 'resources/H.fit'
o_path = 'resources/O.fit'

# 원본 이미지 SHO로 합성
original_image = Processor.create_sho_image(s_path, h_path, o_path)

# 밝기 살짝 조정
brightness_factor = 200  # 밝기 조정 인자
rgb_brightened = Processor.adjust_brightness(original_image, brightness_factor)

# 대비 살짝 조정
contrast_factor = 1.2   # 대비 조정 인자
rgb_contrasted = Processor.adjust_contrast(rgb_brightened, contrast_factor)

# 색상 균형 조정
auto_balanced = Processor.match_histograms(rgb_contrasted)

# asinh 스트레칭 적용
none_linear_factor = 100  # 스트레칭의 강도를 조절할 인자
asinh_stretched = Processor.asinh_stretch(auto_balanced, none_linear_factor)

# 자동 스트레칭
auto_strectched = Processor.auto_stretch(auto_balanced)

# 밝기 조정
rgb_brightened2 = Processor.adjust_brightness(auto_strectched, 0.8)

# 대비 조정
rgb_contrasted2 = Processor.adjust_contrast(rgb_brightened2, 5)

auto_color_balance = Processor.auto_color_balance(rgb_contrasted2)

# 별 제거
rgb_starless = Processor.remove_stars(rgb_contrasted2)

# 렌더링
renderer = Renderer()
renderer.render(rgb_starless)

# TIFF 파일로 저장
renderer.save_as_tif(rgb_starless, 'result.tif')