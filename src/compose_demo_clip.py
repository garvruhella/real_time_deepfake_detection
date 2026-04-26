import os, cv2, argparse, random, numpy as np

def imgs_to_video(imgs, writer, fps=10):
    for im in imgs:
        writer.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--faces_root", default="datasets/detector_data/test")
    parser.add_argument("--out", default="demo_test_video.mp4")
    args = parser.parse_args()

    real_dir = os.path.join(args.faces_root,"real")
    fake_dir = os.path.join(args.faces_root,"fake")
    reals = [cv2.imread(os.path.join(real_dir,f)) for f in os.listdir(real_dir)[:2]]
    fakes = [cv2.imread(os.path.join(fake_dir,f)) for f in os.listdir(fake_dir)[:2]]
    frames = []
    for arr,label in [(reals,'real'),(fakes,'fake')]:
        for img in arr:
            if img is None: continue
            img = cv2.resize(img,(256,256))
            cv2.putText(img,label.upper(),(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0) if label=='real' else (0,0,255),2)
            for _ in range(20): frames.append(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.out, fourcc, 10, (256,256))
    imgs_to_video(frames, out)
    out.release()
    print(f"🎞️ Created {args.out} with {len(frames)} frames")

if __name__=="__main__":
    main()
