# entry_conditions.py
# -------------------
# קובץ זה מגדיר תנאי כניסה להשקעה על בסיס ביטא אסימטרית.
# מטרתו לזהות מניות ש"חזקות מהשוק" ולדרג אותן לפי עוצמה יחסית.
# entry_conditions.py
# -------------------
# 📌 מטרה:
# קובץ זה מגדיר פונקציות לזיהוי מניות "חזקות מהשוק" על בסיס טבלת הביטא הא-סימטרית `asymmetric_betas` במסד הנתונים.
#
# 🗂️ קלט:
# - ערכים מתוך טבלת `asymmetric_betas`, כולל:
#     • symbol
#     • date
#     • beta_up (בטא של המניה בימים שבהם השוק עלה)
#     • beta_down (בטא של המניה בימים שבהם השוק ירד)
#
# ⚙️ מה הקובץ עושה:
# - מוודא אם מניה נחשבת חזקה מהשוק לפי קריטריונים מוגדרים:
#     • β⁺ > 1.1 (המניה עולה יותר מהשוק)
#     • β⁻ > -0.5 (המניה יורדת פחות מהשוק)
# - מחשב ציון חוזקה יחסי (לא חובה): beta_up - abs(beta_down)
#
# 🧾 פלט:
# - bool: האם המניה נחשבת חזקה מהשוק (`True` או `False`)
# - float (אופציונלי): מדד חוזקה יחסית לשוק, להמשך דירוג
#
# 🧠 שימושים:
# - שלב סינון מניות לאסטרטגיות השקעה
# - תשתית לבניית תנאי כניסה לאלגוריתם מסחר
# - דירוג מניות לצורכי ניתוח עומק או אימון מודלים

def is_strong_stock(beta_up, beta_down, beta_up_threshold=1.1, beta_down_threshold=-0.5):
    """
    בודק האם מניה נחשבת חזקה מהשוק:
    - בטא חיובית גבוהה (ביום שהשוק עלה)
    - בטא שלילית נמוכה (ביום שהשוק ירד)

    Args:
        beta_up (float): ביטא בימים חיוביים של השוק
        beta_down (float): ביטא בימים שליליים של השוק
        beta_up_threshold (float): רף תחתון ל־β⁺
        beta_down_threshold (float): רף עליון ל־β⁻

    Returns:
        bool: True אם המניה חזקה מהשוק
    """
    return (
        beta_up is not None and
        beta_down is not None and
        beta_up > beta_up_threshold and
        beta_down > beta_down_threshold
    )


def relative_strength_score(beta_up, beta_down):
    """
    מחשב מדד חוזקה יחסית של מניה לשוק
    (גבוה יותר → המניה חזקה יותר מהשוק)

    Args:
        beta_up (float): בטא כשהשוק עולה
        beta_down (float): בטא כשהשוק יורד

    Returns:
        float: מדד חוזקה יחסית
    """
    if beta_up is None or beta_down is None:
        return None
    return beta_up - abs(beta_down)
