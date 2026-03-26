
import os
import sys

import argparse
import geojson
import numpy as np
import pdnl_sana.geo
import pdnl_sana.interpolate
import pdnl_sana.slide
import pdnl_sana.logging
import pdnl_sana.image
from phas.client.api import Client, Task, Slide, SamplingROITask, DLTrainingTask

from matplotlib import pyplot as plt

SEG_CLASSES = ['CSF', 'R', 'GM', 'L']

def read_geojson(f):
    annotations = []
    data = geojson.load(open(f, 'r'))
    for ann in data['features']:
        xy = np.squeeze(np.array(ann['geometry']['coordinates']))
        cls = ann['properties']['classification']['name']
        annotation = pdnl_sana.geo.Annotation(*xy.T, class_name=cls, level=0)

        # shape checking
        if (len(annotation.shape) != 2) or \
           (annotation.shape[0] < 2) or \
           (annotation.shape[1] != 2):
            print(f"ERROR: Improper polygon -- {annotation.shape} | {annotation.class_name}")
            exit()
        
        annotations.append(annotation)
        
    return annotations

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['local', 'api'])
    parser.add_argument('-s', '--slide', type=str, help="path to slidename, or slide ID")
    parser.add_argument('-a', '--annotation', type=str, help="path to geojson annotation, or annotation ID")
    parser.add_argument('-o', '--output_directory', type=str, help="path to write output data to")
    parser.add_argument('-l', '--level', help="resolution to load the image data at", default=None, type=int)
    parser.add_argument('--url', help='URL path to the API')
    parser.add_argument('--api_key', help='API key providing user level access')
    parser.add_argument('--project_id', help='API project name')
    parser.add_argument('--task_id', help='API task name')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    logger = pdnl_sana.logging.Logger('normal', os.path.join(args.output_directory,'log.pkl'))

    # extract the PNG from the slide file
    if args.mode == 'local':

        # cmdl parsing
        if not os.path.exists(args.slide):
            print(f'ERROR: Slide does not exist -- {args.slide}')
            exit()
        if not os.path.exists(args.annotation):
            print(f'ERROR: Annotation does not exist -- {args.annotation}')
            exit()

        # load the polygons
        annotations = read_geojson(args.annotation)

        # basic ROI loading
        if len(annotations) == 1:
            segments = None
            roi = annotations[0].to_polygon()

        # GM SEG loading
        elif len(annotations) == 4 and all(map(lambda x: x in classes, SEG_CLASSES)):
            classes = [x.class_name for x in annotations]            
            segments = [[x.to_curve() for x in annotations if x.class_name == cls][0] for cls in SEG_CLASSES]
            segments = pdnl_sana.interpolate.clip_quadrilateral_segments(*segments)
            roi = pdnl_sana.geo.connect_segments(*segments)
            
        else:
            print(f'ERROR: Unrecognized annotation format -- {"|".join([(x.class_name, x.annotation_name) for x in annotations])}')
            exit()

        # prepare slide I/O
        loader = pdnl_sana.slide.Loader(logger, args.slide)

        # pixel resolution
        if args.level is None:
            level_ds = ', '.join([f'[level={level}|ds={loader.ds[level]}]' for level in range(len(loader.ds))])
            print(f'INFO: Available levels: {loader.mpp} -- {level_ds}')
            exit()
        else:
            level = args.level

        # transform the polygons into the correct resolution
        if segments:
            [loader.converter.rescale(x, level) for x in segments]
        loader.converter.rescale(roi, level)

        # slide I/O
        frame = loader.load_frame_with_roi(roi, level=level)

        # transform the polygons into the frame coordinate system
        if segments:
            [pdnl_sana.geo.transform_array_with_logger(x, logger) for x in segments]
        pdnl_sana.geo.transform_array_with_logger(roi, logger)

        # create a mask based on the user's annotation
        mask = pdnl_sana.image.create_mask_like(frame, [roi])

        # plotting
        if args.debug:
            fig, axs = plt.subplots(1,2)
            ax = axs[0]
            axs[0].imshow(frame.img)
            if segments:
                [axs[0].plot(*x.T) for x in segments]
        else:
            ax = None

        # cortical deformation
        if not segments is None:
            sample_grid, _ = pdnl_sana.interpolate.fan_sample(*segments, ax=ax)
            if args.debug:
                deform = pdnl_sana.interpolate.grid_sample(frame, sample_grid)
                axs[1].imshow(deform.img)
        else:
            sample_grid = None    

    # extract the PNG using the API
    else:
        
        # connect to the API
        if args.api_key is None or not os.path.exists(args.api_key):
            print('ERROR: Provide proper api_key.json -- {args.api_key}')
            exit()
            
        conn = Client(args.url, args.api_key)
        print(conn)
        if args.project_id is None:
            for project in conn.project_listing():
                print(project)
            exit()
        if args.task_id is None:
            for task in conn.task_listing(args.project_id):
                print(task)
            exit()
        task = Task(conn, args.task_id)

        if args.slide is None:
            print('ERROR: Slide ID required, you must first query for Slide ID')

        if task.detail["mode"] == "sampling":
            task = SamplingROITask(conn, args.task_id)
            rois = task.slide_sampling_rois(args.slide)
        elif task.detail["mode"] == "dltrain":
            task = DLTrainingTask(conn, args.task_id)
            rois = task.slide_training_samples(args.slide)
        else:
            slide = Slide(task=task, slide_id=args.slide)
            exit()
            
        # TODO: replace this w/ bounding box of the polygon
        # TODO: probably extract all ROIs within the slide? and auto-create ROI subdir
        roi = rois[100]
        loc = pdnl_sana.geo.Point(roi['x0'], roi['y0'], is_micron=False, level=0)
        size = pdnl_sana.geo.point_like(loc, roi['x1'], roi['y1']) - loc

        slide = Slide(task=task, slide_id=args.slide)
        mpp = 1000*float(slide.spacing[0])
        ds = slide.level_downsamples
        converter = pdnl_sana.geo.Converter(mpp=mpp, ds=ds)
        if args.level is None:
            level_ds = ', '.join([f'[level={level}|ds={ds[level]}]' for level in range(len(ds))])
            print(f'INFO: Available levels: {mpp} -- {level_ds}')

        ctr = tuple(map(int, loc+size//2))
        converter.rescale(loc, args.level)
        converter.rescale(size, args.level)
        sze = tuple(map(int, size))
        image = slide.get_patch(center=ctr, level=args.level, size=sze)

        logger.data["level"] = args.level
        logger.data["mpp"] = mpp
        logger.data["ds"] = ds
        logger.data["loc"] = loc
        logger.data["size"] = size
        logger.data["padding"] = 0

        frame = pdnl_sana.image.Frame(np.asarray(image), converter=converter, level=args.level)
        
        roi = pdnl_sana.geo.rectangle_like(loc, loc, size)
        roi = pdnl_sana.geo.transform_array_with_logger(roi, logger) 
        mask = pdnl_sana.image.create_mask_like(frame, [roi])

        if args.debug:
            fig, axs = plt.subplots(1,2)
            axs[0].imshow(frame.img)
            axs[0].plot(*roi.T)
            axs[1].imshow(mask.img)

        # TODO: smooth polygon and save deformation curve
        sample_grid = None

    # show the plots
    if args.debug:
        plt.show()

    # save the results!
    else:
        os.makedirs(args.output_directory, exist_ok=True)
        frame.save(os.path.join(args.output_directory, 'frame.png'))
        mask.save(os.path.join(args.output_directory, 'mask.png'))
        logger.write_data()
        if not sample_grid is None:
            np.save(os.path.join(args.output_directory, 'deform.npy'), sample_grid)

if __name__ == "__main__":
    main()
