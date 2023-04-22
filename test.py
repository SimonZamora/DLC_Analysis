import deeplabcut
config_path =r'D:\Simon\Analyses\449_FirstTry\449_FirstTry-SimonZ-2023-04-04\config.yaml'

video_file = r'D:\Simon\Analyses\449_FirstTry\FirstTryTest\Trial_3_449_2023-02-08_16-53-13_camC_TEC_1_36C.avi'
deeplabcut.extract_outlier_frames(config_path, [video_file], outlieralgorithm='manual')


##

