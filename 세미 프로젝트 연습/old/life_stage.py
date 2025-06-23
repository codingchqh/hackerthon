# 6가지 테마를 상수로 정의해두면 코드 관리가 편합니다.
THEMES = [
    "우리의 평범하지만 소중한 일상",
    "함께 떠났던 즐거운 여행의 추억",
    "특별한 날의 행복했던 순간들 (생일, 명절 등)",
    "아이들의 사랑스러운 성장 기록",
    "다시 봐도 웃음이 나는 우리 가족의 재미있는 순간",
    "서로에게 전하는 사랑과 감사의 메시지"
]

def generate_family_story_prompt(family_name: str, theme: str) -> str:
    """
    가족 이름과 테마를 입력받아, 영상 생성 AI를 위한 상세한 영어 프롬프트를 생성합니다.
    """
    
    # 테마별로 최적화된 영어 프롬프트 템플릿 딕셔너리
    prompt_templates = {
        THEMES[0]: f"Create a warm and emotional video showcasing the small, happy moments in the daily life of the '{family_name}' family. Focus on scenes like having breakfast together, a gentle walk in the park, and cozy evening conversations. Use a soft, cinematic style with warm lighting.",
        
        THEMES[1]: f"Generate a cheerful and upbeat video montage of the '{family_name}' family's travel memories. Feature scenes of them laughing against beautiful landscapes, exploring a new city, and enjoying activities together. Use dynamic cuts and an energetic, happy soundtrack.",
        
        THEMES[2]: f"Produce a festive and joyful video celebrating special occasions for the '{family_name}' family. Include clips from a birthday party with cake and candles, a holiday gathering with decorations, and other anniversary celebrations. The mood should be full of happiness and togetherness.",
        
        THEMES[3]: f"Create a touching video documenting the growth of the children in the '{family_name}' family. Show a chronological progression of heartwarming moments: a baby's first steps, the first day of school, learning to ride a bike. The style should be sentimental and nostalgic.",
        
        THEMES[4]: f"Make a funny and comedic video compilation of the '{family_name}' family's hilarious moments, like funny mistakes, practical jokes, and candid bloopers. Edit it like a sitcom opening with laugh tracks and quick cuts.",
        
        THEMES[5]: f"Generate a deeply moving video with a message of love and gratitude from the '{family_name}' family members to each other. Feature close-up shots of family members speaking, with their heartfelt messages displayed as elegant text overlays. Use soft, poignant background music."
    }
    
    # 선택된 테마에 해당하는 프롬프트를 반환, 만약 없는 테마면 기본 프롬프트 반환
    return prompt_templates.get(theme, f"A video about the '{family_name}' family.")

# --- 테스트용 코드 ---
if __name__ == '__main__':
    my_family_name = "Happy Lee Family"

    print("--- 1. '일상' 테마 테스트 ---")
    # Streamlit에서 사용자가 '우리의 평범하지만 소중한 일상'을 선택했다고 가정
    selected_theme_1 = THEMES[0] 
    prompt_1 = generate_family_story_prompt(my_family_name, selected_theme_1)
    print(f"입력 테마: {selected_theme_1}")
    print(f"생성된 프롬프트: {prompt_1}\n")

    print("--- 2. '여행' 테마 테스트 ---")
    # Streamlit에서 사용자가 '함께 떠났던 즐거운 여행의 추억'을 선택했다고 가정
    selected_theme_2 = THEMES[1]
    prompt_2 = generate_family_story_prompt(my_family_name, selected_theme_2)
    print(f"입력 테마: {selected_theme_2}")
    print(f"생성된 프롬프트: {prompt_2}\n")