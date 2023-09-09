import os
import glob
from tqdm.auto import tqdm
import config
"""
visdrone 데이터셋은 다음과 같은 트리구조를 가집니다.
train
    annotations
        각 비디오 별 annotation이 담긴 txt 파일
    sequences
        각 비디오 별 폴더
            프레임 별 jpg 이미지 파일(0000001.jpg)
val
    train과 동일
test
    train과 동일

여기서 0000001.jpg 이미지의 이름을 uav0000013_00000_v_0000001.jpg 과 같이 비디오 이름_프레임 으로 설정합니다.

이유 : 이미지끼리 동일한 이름이 나타나는 것을 방지
"""

if __name__ == "__main__":
    dtypes = ["train","val","test"]
    for dtype in dtypes:
        print(f"{dtype} imagefile rename")
        video_names = glob.glob(os.path.join(config.DATAPATH,f"VisDrone2019-MOT-{dtype}\sequences\*"))
        for vn in tqdm(video_names):
            video_name = vn.split("\\")[-1]
            file_names = os.listdir(vn)
            for file_name in file_names:
                os.rename(
                    vn+"\\"+file_name,
                    vn+"\\"+video_name+"_"+file_name
                )
