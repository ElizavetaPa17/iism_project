# Динамическая верификация подписей

## Шаги для запуска проекта (Windows):

1. Создание виртуальной среды ***venv***:
```
python -m venv venv
```

2. Активация созданной среды:
```
.\venv\Scripts\Activate.ps1
```

**_ВНИМАНИЕ:_** Если на шаге 2 возникла ошибка вида ".\venv\Scripts\Activate.ps1 : Невозможно загрузить файл D:\iism_project\venv\Scripts\Activate.ps1, так как выполнение
сценариев отключено в этой системе.", необходимо:
 - Открыть терминал PowerShell от админа.
 - Вставить и запустить - Set-ExecutionPolicy RemoteSigned.
 - На вопрос ответить - A.
 - Вернуться к шагу 1, <ins>открыв новый терминал</ins>.


3. Установка зависимостей:
```
python -m pip install -r libs.txt
```

4. Запуск GUI:
```
python main.py
```
