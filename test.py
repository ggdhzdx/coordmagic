import coordmagic as cm
st = cm.read_structure('NI-6NH3.log')
cm.write_structure(st,'test.mol2', connection=True)
