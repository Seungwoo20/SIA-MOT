# Data 전처리 과정
1. 다운 받은 Visdrone 데이터셋에서 test-dev 폴더명을 test로 바꾼다.
2. eda/config.py 의 파일의 DATAPATH 를 데이터 셋이 있는 폴더로 설정한다.
3. 각 비디오의 파일명이 동일한 것을 방지하기 위해 이름을 변경한다.
    * change_filename.py 실행
4. Annotation에서 Detection 용 데이터 프레임을 생성한다.
    * make_det_annotation.py 실행
        1. dataframe.csv 는 Annotation 파일을 읽어 객체별로 분리한 내용
        2. detection_frame.csv 는 각 프레임(이미지)별로 객체를 취합한 내용
5. Detection 훈련 또는 추론을 위한 tf.record 파일 생성
    * make_tfrecord.py 실행