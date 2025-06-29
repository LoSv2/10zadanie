# Система преобразования координат

Это веб-приложение предназначено для перевода координат между различными системами, такими как СК-42, СК-95 и другими.

##  Развернутое приложение

Фронтенд (Streamlit):  
https://10zadanie-hrj39d7rx759dsyfvlfdkg.streamlit.app/


---


## Структура проекта

```
.
├── main.py
├── front/
│   └── app.py
├── requirements.txt
├── parameters.json
└── README.md
```

## Функционал

- Загрузка Excel-файлов с координатными данными
- Преобразование координат с настраиваемыми параметрами
- Генерация отчетов в формате markdown
- Скачивание преобразованных данных и отчетов
- Современный веб-интерфейс на Streamlit

## Инструкции по установке

### Локальная разработка

1. Клонируйте репозиторий:
```bash
git clone <url-репозитория>
cd coordinate-transformation-system
```

2. Создайте виртуальное окружение и установите зависимости:
```bash
python -m venv venv
source venv/bin/activate  # Для Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Запустите бэкенд-сервер:
```bash
cd backend
uvicorn backend.main:app --reload
```

4. Запустите фронтенд-приложение:
```bash
cd frontend
streamlit run app.py
```

### Развертывание

#### Бэкенд (Render.com)

1. Создайте новый Web Service на Render.com
2. Подключите ваш GitHub репозиторий
3. Установите следующие параметры:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. Разверните сервис

#### Фронтенд (Streamlit Cloud)

1. Создайте аккаунт на Streamlit Cloud
2. Подключите ваш GitHub репозиторий
3. Установите следующие параметры:
   - Main file path: `front/app.py`
   - Python version: 3.10
4. Обновите `BACKEND_URL` в `front/app.py`, указав URL вашего развернутого бэкенда
5. Разверните приложение

## Как пользоваться

1. Откройте Streamlit приложение в браузере
2. Загрузите Excel-файл с координатными данными (столбцы: x, y, z)
3. Настройте параметры преобразования
4. Нажмите "Преобразовать координаты"
5. Скачайте результаты
