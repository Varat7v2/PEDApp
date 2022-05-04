import rawpy

# with open ('data/UH_ECD.raw') as f:
# 	for line in f:
# 		print(line.strip())


raw = rawpy.imread('data/UH_ECD.raw')
print(raw.postprocess())