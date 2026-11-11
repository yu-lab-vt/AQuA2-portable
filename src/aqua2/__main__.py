import argparse
import json
from pathlib import Path

from .compare import Tolerance, compare
from .io import read_mat, write_mat


def main():
    parser = argparse.ArgumentParser(description="AQuA2 event detection, scientific features and MATLAB parity tools")
    commands = parser.add_subparsers(dest="command", required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("reference", type=Path)
    comparison.add_argument("actual", type=Path)
    comparison.add_argument("--field", action="append", help="Optional MAT field or slash path")
    comparison.add_argument("--atol", type=float, default=0.0)
    comparison.add_argument("--rtol", type=float, default=0.0)
    comparison.add_argument("--output", type=Path)
    processing = commands.add_parser("preprocess",help="Run input preparation, baseline/noise and active regions")
    processing.add_argument("input",type=Path)
    processing.add_argument("output",type=Path)
    processing.add_argument("--parameters",type=Path)
    processing.add_argument("--preset",type=int,default=1)
    processing.add_argument("--channel2",type=Path)
    processing.add_argument("--variable")
    processing.add_argument("--mask",type=Path)
    processing.add_argument("--saved-result",action="store_true")
    processing.add_argument("--gui-median",action="store_true",help="Enable the GUI median step; batch default omits it")
    processing.add_argument("--frames",help="Zero-based frame slice, e.g. 0:100")
    detection = commands.add_parser("detect",help="Run preprocessing, temporal/spatial segmentation and optional global detection")
    for action in processing._actions:
        if action.dest!="help":detection._add_action(action)
    detection.add_argument("--no-features",action="store_true",help="Stop after stage-5 detection")
    detection.add_argument("--regions",type=Path,help="MAT regions cell list, in working coordinates")
    detection.add_argument("--landmarks",type=Path,help="MAT landmarks cell list, in working coordinates")
    detection.add_argument("--export-dir",type=Path,help="Export feature/curve CSV and numeric maps")
    features = commands.add_parser("features",help="Extract features from stage-3 and stage-5 MAT outputs")
    features.add_argument("input",type=Path,help="Stage-3 MAT with datOrg/opts")
    features.add_argument("events",type=Path,help="Detection MAT with channel1/local, optional global and channel2")
    features.add_argument("output",type=Path)
    features.add_argument("--regions",type=Path)
    features.add_argument("--landmarks",type=Path)
    features.add_argument("--export-dir",type=Path)
    temporal = commands.add_parser("temporal",help="Run temporal segmentation on a stage-3 MAT output")
    temporal.add_argument("input",type=Path)
    temporal.add_argument("output",type=Path)
    cfu = commands.add_parser("cfu",help="CFU detection/dependency/grouping from a saved AQuA2 res MAT")
    cfu.add_argument("input",type=Path)
    cfu.add_argument("output",type=Path)
    cfu.add_argument("--options",type=Path,help="JSON CFU options or MAT containing cfuOpts")
    cfu.add_argument("--max-distance",type=int)
    cfu.add_argument("--shift",type=int)
    run = commands.add_parser('run',help='Integrated detection, features, CFU and a reopenable Python project')
    run.add_argument('input',type=Path);run.add_argument('output',type=Path,nargs='?')
    run.add_argument('--save-project',action='store_true',help='Explicitly save the computed project to output')
    run.add_argument('--export-dir',type=Path)
    run.add_argument('--parameters',type=Path);run.add_argument('--preset',type=int,default=1)
    run.add_argument('--cfu-options',type=Path);run.add_argument('--no-cfu',action='store_true')
    run.add_argument('--saved-result',action='store_true');run.add_argument('--channel2',type=Path)
    run.add_argument('--variable');run.add_argument('--frames');run.add_argument('--mask',type=Path)
    run.add_argument('--regions',type=Path);run.add_argument('--landmarks',type=Path)
    run.add_argument('--overwrite',action='store_true')
    batch = commands.add_parser('batch',help='Run a JSON batch manifest with per-job status')
    batch.add_argument('manifest',type=Path);batch.add_argument('--report',type=Path)
    importing=commands.add_parser('import-project',help='Import saved MATLAB events/features and optional CFUs')
    importing.add_argument('input',type=Path);importing.add_argument('output',type=Path);importing.add_argument('--cfu-file',type=Path);importing.add_argument('--embed-movie',action='store_true')
    for name in ('gui','view'):
        viewer=commands.add_parser(name,help='Open the local AQuA2 desktop workbench')
        viewer.add_argument('project',type=Path,nargs='?')
        viewer.add_argument('--image-backend',choices=('auto','tk','qt-gpu'),default='auto',help='Image display backend; auto tries GPU and falls back to Tk')
    rendering=commands.add_parser('render',help='Export an event/CFU scientific preview without opening a GUI')
    rendering.add_argument('project',type=Path);rendering.add_argument('output',type=Path)
    rendering.add_argument('--cfu',action='store_true');rendering.add_argument('--channel',type=int,choices=[1,2],default=1)
    rendering.add_argument('--frame',type=int,default=1,help='Displayed one-based frame number')
    rendering.add_argument('--ids',default='1',help='Displayed one-based IDs, comma separated');rendering.add_argument('--mean',action='store_true')
    worker=commands.add_parser('_worker',help=argparse.SUPPRESS);worker.add_argument('job',type=Path)
    args = parser.parse_args()
    if args.command in ('gui','view'):
        try:from .viewer import launch
        except ImportError as error:parser.error('Desktop viewer needs Tk and matplotlib; install aqua2[viewer]. '+str(error))
        launch(args.project,image_backend=args.image_backend);return 0
    if args.command=='render':
        from .visualization import export_preview
        path=export_preview(args.project,args.output,mode='CFU' if args.cfu else 'Event',channel=f'channel{args.channel}',frame=args.frame-1,
            selected=tuple(int(v)-1 for v in args.ids.split(',')),projection=args.mean)
        print(json.dumps({'preview':str(path)}));return 0
    if args.command=='_worker':
        from .workflow import execute_job
        import traceback,threading,time
        stopped=threading.Event();output_lock=threading.Lock()
        def send(message):
            with output_lock:print(json.dumps(message),flush=True)
        def heartbeat():
            while not stopped.wait(5):send({'type':'heartbeat','cpu_seconds':time.process_time()})
        threading.Thread(target=heartbeat,daemon=True).start()
        try:
            result=execute_job(json.loads(args.job.read_text(encoding='utf-8')),progress=lambda value:send({'type':'progress',**value}))
            send({'type':'complete',**result});return 0
        except Exception as error:
            traceback.print_exc();send({'type':'error','error':str(error)});return 1
        finally:stopped.set()
    if args.command=='run':
        from .workflow import run_workflow
        config={k:str(getattr(args,k)) for k in ('parameters','cfu_options','mask','regions','landmarks','channel2') if getattr(args,k)}
        config.update(preset=args.preset,with_cfu=not args.no_cfu,saved_result=args.saved_result,variable=args.variable)
        if args.frames:
            parts=args.frames.split(':')
            if not 2<=len(parts)<=3:parser.error('--frames must be start:stop[:step]')
            config['frames']=[int(p) if p else None for p in parts]
        project=run_workflow(args.input,args.output,config,save_project=args.save_project,overwrite=args.overwrite,
            progress=lambda value:print(json.dumps({'type':'progress',**value}),flush=True))
        if args.export_dir:
            from .project import export_project
            export_project(project,args.export_dir)
        print(json.dumps({'project':str(project.path) if project.path else None,**project.summary()}));return 0
    if args.command=='batch':
        from .workflow import run_batch
        result=run_batch(args.manifest,progress=lambda value:print(json.dumps(value),flush=True))
        text=json.dumps(result,ensure_ascii=False,indent=2)
        if args.report:args.report.write_text(text+'\n',encoding='utf-8')
        print(text);return 0 if result['passed'] else 1
    if args.command=='import-project':
        from .project import import_saved_project
        project=import_saved_project(args.input,args.output,cfu_file=args.cfu_file,embed_external=args.embed_movie)
        print(json.dumps({'project':str(project.path),**project.summary()}));return 0
    if args.command=="cfu":
        from .cfu import cfu_from_saved_result,cfu_mat_payload,default_options
        settings=default_options()
        if args.options:
            settings=json.loads(args.options.read_text(encoding='utf-8')) if args.options.suffix.lower()=='.json' else read_mat(args.options,['cfuOpts'])['cfuOpts']
        if args.max_distance is not None:settings['cfuAnalysis']['maxDist']=args.max_distance
        if args.shift is not None:settings['cfuAnalysis']['shift']=args.shift
        result=cfu_from_saved_result(args.input,settings)
        write_mat(args.output,cfu_mat_payload(result))
        print(json.dumps({'output':str(args.output),'cfus':[len(result['cfuInfo1']),len(result['cfuInfo2'])],'relations':len(result['cfuRelation']),'groups':len(result['cfuGroupInfo'])}));return 0
    if args.command=="features":
        from .se import read_stage3_payload
        from .evt import spatial_from_mat_payload
        from .features import compute_features,mask_options_from_files
        from .fea import features_mat_payload
        from .export import export_features
        stage3=read_stage3_payload(args.input);events=read_mat(args.events)
        masks=mask_options_from_files(args.regions,args.landmarks);payload={"opts":stage3["opts"]}
        for channel in (1,2):
            key=f"channel{channel}"
            if key not in events:continue
            payload[key]={}
            for name,stored in events[key].items():
                spatial=spatial_from_mat_payload(stored,stage3["opts"])
                if name=="global" and not len(spatial["evt"]):continue
                result=compute_features(stage3[f"datOrg{channel}"],spatial,stage3["opts"],channel=channel,**masks)
                payload[key][name]=features_mat_payload(result)
                if args.export_dir:export_features(args.export_dir,result,prefix=f"channel{channel}_{name}",rising_maps=spatial["riseLst"])
        write_mat(args.output,payload);print(json.dumps({"output":str(args.output),"export_dir":str(args.export_dir) if args.export_dir else None}));return 0
    if args.command == "temporal":
        from .se import read_stage3_payload,temporal_from_stage3,temporal_mat_payload
        stage3 = read_stage3_payload(args.input)
        channels = temporal_from_stage3(stage3)
        if not channels:
            parser.error("Input has no dF1 or dF2 stage-3 data")
        payload = {name:temporal_mat_payload(result) for name,result in channels.items()}
        payload["opts"] = next(iter(channels.values()))["opts"]
        write_mat(args.output,payload)
        print(json.dumps({"output":str(args.output),"channels":{
            name:{"sub_events":len(value["subEvtLst"]),"super_events":len(value["seLst"])}
            for name,value in channels.items()}}))
        return 0
    if args.command in ("preprocess","detect"):
        from .config import read_preset
        from .pipeline import preprocess,stage3_mat_payload
        options = None
        if args.parameters is not None or not args.saved_result:
            parameters = args.parameters or Path(__file__).with_name("data") / "parameters_for_batch.csv"
            options = read_preset(parameters,args.preset)
        frames = None
        if args.frames:
            parts = args.frames.split(":")
            if not 2 <= len(parts) <= 3:
                parser.error("--frames must be start:stop[:step]")
            frames = slice(*(int(part) if part else None for part in parts))
        result = preprocess(args.input,options,args.channel2,variable=args.variable,
            spatial_mask=args.mask,saved_result=args.saved_result,frames=frames,gui_median=args.gui_median)
        if args.command=="detect":
            from .detection import detect_from_stage3,detection_mat_payload
            from .io import read_working_mask
            mask=read_working_mask(args.mask,result["dF1"].shape[:3],result["opts"].get("regMaskGap",0)) if args.mask else None
            from .features import mask_options_from_files
            if args.no_features and args.export_dir:parser.error("--export-dir requires features")
            detected=detect_from_stage3(result,spatial_mask=mask,with_features=not args.no_features,**mask_options_from_files(args.regions,args.landmarks))
            write_mat(args.output,detection_mat_payload(detected))
            if args.export_dir:
                from .export import export_features
                for key,value in detected.items():
                    if key=="opts":continue
                    for name,spatial in value.items():
                        if "features" in spatial:export_features(args.export_dir,spatial["features"],prefix=f"{key}_{name}",rising_maps=spatial["riseLst"])
            print(json.dumps({"output":str(args.output),"channels":{
                key:{"events":len(value["local"]["evt"]),"global_events":len(value["global"]["evt"])}
                for key,value in detected.items() if key!="opts"}}))
            return 0
        write_mat(args.output,stage3_mat_payload(result))
        print(json.dumps({"output":str(args.output),"shape_hwlt":list(result["datOrg1"].shape),
            "active_regions_channel1":len(result["activeRegions1"]),
            "validation_note":"Independent 3D/dual-channel/registration/bleaching/mask references deferred to step 9"}))
        return 0
    result = compare(read_mat(args.reference, args.field), read_mat(args.actual, args.field),
                     tolerance=Tolerance(args.atol, args.rtol))
    report = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
