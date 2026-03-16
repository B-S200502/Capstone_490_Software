Import("env")

# Force the same flags for the *link step*
env.Append(LINKFLAGS=[
    "-mfpu=fpv4-sp-d16",
    "-mfloat-abi=hard",
])