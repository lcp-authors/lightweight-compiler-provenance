'''
distinct-opcodes-registers.py

Records the number of distinct opcodes used and the
number of distinct general-purpose registers used by the
main function of each binary.
'''

import r2pipe
import sys
import os
import multiprocessing as mp

# These variables will be set by main, depending on command line args
registers = None
profileRegisters = None


'''
Yield successive n-sized chunks from lst.
Source: https://stackoverflow.com/questions
/312443/how-do-you-split-a-list-into-evenly-sized-chunks
'''
def chunks(lst, n):

    for i in range(0, len(lst), n):
        yield lst[i:i + n]


'''
This function runs in parallel to disassemble the main function
of a binary and record the number of unique opcodes or registers
that main() uses.
'''
def worker(basenames, outfile, num):
    assert(registers is not None)
    assert(profileRegisters is not None)

    outfile = open(str(num) + "-" + outfile, 'a')

    for bname in basenames:
        path = os.path.join(sys.argv[1], bname)

        # Connect to radare2. Commands stand for analyze all (aaa),
        # then choose symbol (s) "main", then print disassembly in
        # JSON format (pdfj).
        r = r2pipe.open(path)
        r.cmd("aaa; s main")
        out = r.cmdj("pdfj")
        
        unique_data = set()

        for instr_dict in out['ops']:
            if "opcode" not in instr_dict:
                continue
            
            instr = instr_dict['opcode']
            opcode = instr.split(" ")[0]

            # If the metric is opcode, the rest of the instruction
            # can be ignored
            if not profileRegisters:
                unique_data.add(opcode)
                continue

            temp = instr.strip().replace("[", "")
            temp = temp.replace("]", "")
            temp = temp.replace("{", "")
            temp = temp.replace("}", "")
            temp = temp.replace(",", "")
            splits = temp.split(" ")

            # If an operand is a general-purpose register, add to set
            for token in splits:
                if token in registers:
                    unique_data.add(token)

        outfile.write(f"{bname} {len(unique_data)}\n")
    
    outfile.close()


'''
Bootstraps parallel CFG extraction.
<metric> is either "opcodes" or "registers".
<arch> is either "amd64" or "armhf".
'''
def main():
    if len(sys.argv) != 6:
        print(f"Usage: {sys.argv[0]} <binary directory> <outfile> <# of parallel jobs> <metric> <arch>")
        exit(1)

    assert(os.path.isdir(sys.argv[1]))
    basenames = os.listdir(sys.argv[1])

    NUM_CORES = sys.argv[3]
    split_basenames = list(chunks(basenames, len(basenames) // NUM_CORES + 1))

    if sys.argv[4] == "opcodes":
        profileRegisters = False
    elif sys.argv[4] == "registers":
        profileRegisters = True
    else:
        print("Invalid argument for <metric>")
        exit(1)

    if sys.argv[5] == "amd64":
        registers = ["rax", "rbx", "rcx", "rdx", "rdi", "rsi", "rsp", "rbp", "r8",
                    "r9", "r10", "r11", "r12", "r13", "r14", "r15", "rip"]
    elif sys.argv[5] == "armhf":
        registers = ['r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7',
                    'r8', 'r9', 'sl', 'fp', 'ip', 'sp', 'lr', 'pc',
                    'CPSR']
    else:
        print("Invalid argument for <arch>")
        exit(1)

    processes = [mp.Process(target=worker,
                            args=(split_basenames[i], sys.argv[2], i))
                            for i in range(len(split_basenames))]

    for p in processes:
        p.start()
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
