from pathlib import Path
import common
import numpy as np
import setting
from sklearn.metrics.pairwise import cosine_similarity
import progressbar


def gen_candidate_database():
    from imagefeature import ImageFeature
    print("Candidate Matching Database Generation Start")

    common.prepare_clean_dir(Path("temp/"))

    IF = ImageFeature()

    query_features, query_pathes, orbs = [], [], []
    for img_file in sorted(Path("input/query/").glob("*")):
        query_pathes.append(img_file)
        x = IF.get_feature(img_file)
        query_features.append(x)
        print("Extracting Query Feature", img_file)

    target_class_names, target_features, target_pathes = [], [], []
    for folder in sorted(Path("input/target/").glob("*")):
        class_name = folder.stem
        for img_file in sorted(Path("input/target/%s/" % class_name).glob("*")):
            target_pathes.append(img_file)
            target_class_names.append(class_name)
            feature = IF.get_feature(img_file)
            target_features.append(feature)
            print("Extracting Target Feature", img_file)

    print("Calculating Similarities...")
    sims = cosine_similarity(query_features, target_features)

    candidate_matching_database = {}

    for query_index, row in enumerate(sims):

        query_file = query_pathes[query_index]
        candidate_matching_database[query_file] = {}

        args = np.argsort(row)
        args = args[::-1]

        for arg in args:
            target_path = target_pathes[arg]
            target_class_name = target_class_names[arg]

            if target_class_name not in candidate_matching_database[query_file]:
                candidate_matching_database[query_file][target_class_name] = []

            if len(candidate_matching_database[query_file][target_class_name]) < setting.MAX_NUMBER_ONE_CLASS:
                candidate_matching_database[query_file][target_class_name].append((target_path, row[arg]))

    common.save_pickle(Path("temp/candidate_matching_database.pickle"), candidate_matching_database)

    print("Candidate Matching Database Generation Finish")


def match():
    import deepmatching_wrapper as dm
    import cv2

    candidate_matching_database = common.load_pickle(Path("temp/candidate_matching_database.pickle"))

    common.prepare_clean_dir(Path("output/"))
    common.prepare_clean_dir(Path("output/images/"))

    output = {}
    for query_file, candidates in candidate_matching_database.items():
        query_name = Path(query_file).stem
        matching_result = []
        for target_class_name, target_images in candidates.items():
            for i, (target_path, similarity) in enumerate(target_images):
                print("Matching", query_file, "with target image", target_path)

                matches, name1, name2, qw, qh, tw, th, img1, img2 = dm.match(query_file, target_path)
                src_pts = np.float32([[m[0], m[1]] for m in matches])
                dst_pts = np.float32([[m[2], m[3]] for m in matches])

                i = 0
                inlier = []

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, setting.RANSAC_THRESHOLD)
                for index, m in enumerate(mask):
                    if np.isclose(m, 1):
                        i += 1
                        inlier.append(matches[index])

                output_name = "%s_%s_%02d.jpg" % (query_name, target_class_name, i)
                dm.draw(img1, img2, inlier, Path("output/images/") / output_name)

                matching_result.append({
                    "class_name": target_class_name,
                    "inlier": len(inlier)
                })
        output[query_file.name] = sorted(matching_result, key=lambda x: x["inlier"], reverse=True)

    common.write_json(Path("output/result.json"), output)

def summary():
    print("Summary:")
    result = common.load_json("output/result.json")
    for query_image, target_images in result.items():
        print("Query Image" , query_image , "is probably", target_images[0]["class_name"], "with" , target_images[0]["inlier"], "inlier feature points")

if __name__ == "__main__":
    # gen_candidate_database()
    match()
    summary()
