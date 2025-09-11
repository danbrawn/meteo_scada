# В корена на проекта (където са app.py, templates/, static/, config.ini):
py -m pip install --upgrade pip pyinstaller
pyinstaller --clean --noconfirm --onedir --name MeteoScada `
  --add-data "config.ini;." `
  --add-data "template.xlsx;." `
  --add-data "templates;templates" `
  --add-data "static;static" `
  app.py
