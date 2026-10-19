import nanosurf
import gwyfile
import time
import sys

def main():
    # Studio must be running before executing the following!
    studio = nanosurf.Studio()
    if studio.connect():
        print(f"Available sessions: {studio.main.session.list()}")
        print(f"Connected with session '{studio.session_id}'")


    # Abbreviations to access the workflows used in the script
    imaging = studio.spm.workflow.imaging
    workspace = studio.spm.workflow.workspace
    automation = studio.spm.workflow.automation
## AUTOMATIZATION ------------------------------------------------------------

    tip_pos_x = -392e-6
    tip_pos_y=-359e-6
    size_lr = 20e-6
    resolution_lr = 256
    scan_rate_lr = 5

    resolution_hr = 512
    scan_rate_hr = 2
   
    id_list = []            # lista de id
    index_lr = 0           # contador de imagenes de baja resolucion

    # The workspace is visible in the ViewPort.
    # Clear the workspace
    workspace.delete_all_items()
    
    # How many items are in the workspace? (Default AFM and camera frames + example items in layer 1)
    workspace.item_count()

    # primera imgen de baja resolucion    
    id = workspace.add_frame("lr" + str(index_lr + 1), tip_pos_x, tip_pos_y)

    workspace.set_item_size(id, size_lr, size_lr)
    workspace.set_item_resolution(id, resolution_lr, resolution_lr)
    workspace.set_item_line_rate(id, scan_rate_lr,True)

    # anado el frame
    automation.add_to_queue(id)

    # How many items are in the queue now?
    automation.queue_size()

    # Show all items in the queue as a formatted string. Useful for verification and debugging.
    # Use print() to preserve formatting.
    print(automation.queue_info())

    # The following commands control queue acquisition. 
    # 'Resume' continues at the queue index at which the queue was paused.
    # 'Start' always starts from the beginning.
    automation.start()
    # You can check the status of queue processing
    automation.is_idle()
    automation.is_paused()
    automation.is_running()
    time.sleep(0.5)
    dots = ''
    print('Start automatization')
    while  automation.is_running():
        dots += "."
        if len(dots) > 4:
            dots = ""

        sys.stdout.write(f"\rScanning{dots}   ")
        sys.stdout.flush()

        time.sleep(0.3)

    print('fin automatization')

   
    # cap
## IMAGING -------------------------------------------------------------------
    # # Change the current (global) imaging settings 
    # # Line rate in hertz, scan ranges in metres
    # imaging.property.line_rate.value = 20

    # imaging.property.scan_range_fast_axis.value = 20e-6   # This is the width of the frame
    # imaging.property.scan_range_slow_axis.value = 20e-6   # This is the height of the frame
    # imaging.property.rotation.value = 45                  # This is the angel of the frame

    # # Access value and unit
    # print(imaging.property.scan_range_fast_axis.value)
    # print(imaging.property.scan_range_fast_axis.unit)

    # # Resolution of the frame
    # imaging.property.points_per_line.value = 200
    # imaging.property.lines_per_frame.value = 200

    # # Show the current scanner offsets (with respect to the centre of the frame)
    # print("Offset X: ", imaging.property.image_offset_x)
    # print("Offset Y: ", imaging.property.image_offset_y)

    # # Change offsets
    # imaging.property.image_offset_x.value = -20e-6
    # imaging.property.image_offset_y.value = -20e-6

    # # Start scanning with the current parameters.
    # # Note that because we changed the parameters, a new frame is automatically created in the ViewPort.
    # imaging.start_imaging()
    # time.sleep(0.5)
    # dots = ''
    # print('Start imaging')
    # while imaging.is_scanning():
    #     dots += "."
    #     if len(dots) > 4:
    #         dots = ""

    #     sys.stdout.write(f"\rScanning{dots}   ")
    #     sys.stdout.flush()

    #     time.sleep(0.3)

    # print('fin imaging')


if __name__ == "__main__":
    main()