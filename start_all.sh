sudo python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
sudo nohup python3 ai_check_photo/main.py > ai_check_photo.log 2>&1 &
sudo nohup python3 ai_multy_modal/main.py > ai_multy_modal.log 2>&1 &
sudo nohup python3 ai_text_operation/main.py > ai_text_operation.log 2>&1 &
echo "Все скрипты запущены с правами sudo!"
