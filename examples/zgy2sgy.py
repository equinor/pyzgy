import segyio
import pyzgy

zgy_in = sys.argv[1]
sgy_out = sys.argv[2]

if len(sys.argv) != 3:
    raise RuntimeError("This example accepts exactly 2 arguments: input_file & output_file")

with pyzgy.open(zgy_in) as input_zgy:
    spec = segyio.spec()
    spec.ilines = input_zgy.ilines
    spec.xlines = input_zgy.xlines
    spec.samples = input_zgy.samples
    spec.format = 1
    with segyio.create(sgy_out, spec) as output_sgy:
        output_sgy.header = input_zgy.header
        for il in input_zgy.ilines:
            print(f'il={il}')
            output_sgy.iline[il] = input_zgy.iline[il]
