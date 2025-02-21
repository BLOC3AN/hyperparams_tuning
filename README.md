# hyperparams_tuning

## Tải data CSV từ `Google-Driver` 

Cài đặt `gdown`
```py
pip install gdown
```
Lấy thông tin đường link liên lết:
```
https://drive.google.com/file/d/<code>/view?usp=drive_link
```
Ví dụ
```
https://drive.google.com/file/d/13NNQ8fv7__UBH4vaRKi7J0CZLgtlyV-q/view?usp=drive_link
```
> &rarr; `13NNQ8fv7__UBH4vaRKi7J0CZLgtlyV-q`

Chạy trong terminel
```bash
gdown 13NNQ8fv7__UBH4vaRKi7J0CZLgtlyV-q -O data.csv
```
