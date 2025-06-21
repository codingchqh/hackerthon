def calculate_life_stages(birth_year: int):
    """
    생년을 입력하면 각 연령대에 해당하는 연도를 반환
    """
    stages = {
        "유아기 (0~9세)": birth_year + 5,
        "10대 (10~19세)": birth_year + 15,
        "20대": birth_year + 25,
        "30대": birth_year + 35,
        "40대": birth_year + 45,
        "50대": birth_year + 55,
        "60대": birth_year + 65,
        "70대": birth_year + 75
    }
    return stages

def get_prompt_by_age(birth_year: int, current_age: int, name: str = "이름"):
    """
    현재 나이 기반으로 시나리오용 영어 프롬프트 예시 반환
    """
    base_year = birth_year + current_age
    if current_age < 20:
        return f"A cheerful Korean child named {name}, playing in the late {base_year}s."
    elif current_age < 40:
        return f"A dynamic young adult named {name}, experiencing love and challenges in the {base_year}s."
    elif current_age < 60:
        return f"A mature {current_age}-year-old named {name}, balancing work and family life in the {base_year}s."
    else:
        return f"An experienced person in their {current_age}s named {name}, reflecting on life in the {base_year}s."
