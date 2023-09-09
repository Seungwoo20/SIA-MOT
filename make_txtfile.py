import pandas as pd
import ast
import numpy as np

def activate_list_in_dataframe(df):
    for i in df.columns:
        a = df[i].values[0]
        if type(a) == type("a"):
            if a.startswith("["):
                df[i] = df[i].apply(lambda x: ast.literal_eval(x))
    return df

def make_txtfile(dtype):
    df_path = f"./data/{dtype}_detection_frame.csv"

    df = pd.read_csv(df_path)
    df = activate_list_in_dataframe(df)

    output_path = f"./data/visdrone_{dtype}_33.txt"

    f = open(output_path, 'w')
    for idx in range(len(df)):
        if idx%3==0:
            img_path = df["img_path"].values[idx]
            obj_counts = len(df["x_min"].values[idx])

            target = []
            for obj in range(obj_counts):
                label = []
                label.append(str(df["x_min"].values[idx][obj]))
                label.append(str(df["y_min"].values[idx][obj]))
                label.append(str(df["x_max"].values[idx][obj]))
                label.append(str(df["y_max"].values[idx][obj]))
                label.append(str(df["classes"].values[idx][obj]))
                target.append(",".join(label))
            y = " ".join(target)
            write_data = img_path +" "+y+"\n"
            f.write(write_data)
    f.close()

if __name__ == "__main__":
    # for dtype in ["train","val","test"]:
    for dtype in ["train"]:
        make_txtfile(dtype)