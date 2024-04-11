import os
import sys
from pathlib import Path
from subprocess import Popen, PIPE

os.environ["SWIFTRANK_MODEL"] = "ms-marco-TinyBERT-L-2-v2"
exec_args = [sys.executable, '-m', 'swiftrank.interface.cli']
files_path = Path(__file__).parent.parent / 'files'

def read_file_bytes(name: str):
    return (files_path / name).read_bytes()
    
def test_print_relevant_context():
    process = Popen(
        [*exec_args, '-q', 'Jujutsu Kaisen: Season 2', '-f'], stdin=PIPE, stdout=PIPE, stderr=PIPE
    )
    
    process.stdin.write(read_file_bytes('contexts'))
    process.stdin.close()
    
    stdout, stderr = process.communicate()
    assert stdout.decode().strip() == "Jujutsu Kaisen 2nd Season"    
    assert stderr.decode().strip() == ""

def test_filtering_using_threshold():
    process = Popen(
        [*exec_args, '-q', 'Jujutsu Kaisen: Season 2', '-t', '0.98'], stdin=PIPE, stdout=PIPE, stderr=PIPE
    )
    
    process.stdin.write(read_file_bytes('contexts'))
    process.stdin.close()
    
    stdout, stderr = process.communicate()
    rlist = [i.strip() for i in stdout.decode().split('\n') if i]
    assert rlist == ['Jujutsu Kaisen 2nd Season', 'Jujutsu Kaisen 2nd Season Recaps']
    assert stderr.decode().strip() == ""

def test_handling_json():
    process = Popen(
        [*exec_args, '-q', 'Jujutsu Kaisen: Season 2', 'process', '-r', '.categories[].items', '-c', '.name', '-t', '0.9'], stdin=PIPE, stdout=PIPE, stderr=PIPE
    )
    
    process.stdin.write(read_file_bytes('contexts.json'))
    process.stdin.close()

    stdout, stderr = process.communicate()
    rlist = [i.strip() for i in stdout.decode().split('\n') if i]
    assert rlist == ['Jujutsu Kaisen 2nd Season', 
        'Jujutsu Kaisen 2nd Season Recaps', 
        'Jujutsu Kaisen', 
        'Jujutsu Kaisen Official PV', 
        'Jujutsu Kaisen 0 Movie']
    assert stderr.decode().strip() == ""

def test_handling_json_with_post_processing():
    process = Popen(
        [*exec_args, '-q', 'Jujutsu Kaisen: Season 2', 'process', '-r', '.categories[].items', '-c', '.name', '-p', '.url', '-f'], stdin=PIPE, stdout=PIPE, stderr=PIPE
    )
    
    process.stdin.write(read_file_bytes('contexts.json'))
    process.stdin.close()

    stdout, stderr = process.communicate()   
    assert stdout.decode().strip() == "https://myanimelist.net/anime/51009/Jujutsu_Kaisen_2nd_Season"    
    assert stderr.decode().strip() == ""

def test_handling_yaml_with_post_processing():
    process = Popen(
        [*exec_args, '-q', 'Monogatari Series: Season 2', 'process', '-r', '.categories[].items', '-c', '.name', '-p', '.payload.status', '-f'], stdin=PIPE, stdout=PIPE, stderr=PIPE
    )
    
    process.stdin.write(read_file_bytes('contexts.yaml'))
    process.stdin.close()

    stdout, stderr = process.communicate()   
    assert stdout.decode().strip() == "Finished Airing"    
    assert stderr.decode().strip() == ""

def test_handling_jsonlines_with_post_processing():
    process = Popen(
        [*exec_args, '-q', 'Monogatari Series: Season 2', 'process', '-c', '.name', '-p', '.payload.aired', '-f'], stdin=PIPE, stdout=PIPE, stderr=PIPE
    )
    
    process.stdin.write(read_file_bytes('contexts.jsonl'))
    process.stdin.close()

    stdout, stderr = process.communicate()   
    assert stdout.decode().strip() == "Jul 7, 2013 to Dec 29, 2013"    
    assert stderr.decode().strip() == ""    

def test_handling_yamllines():
    process = Popen(
        [*exec_args, '-q', 'Monogatari Series: Season 2', 'process', '-c', '.name', '-f'], stdin=PIPE, stdout=PIPE, stderr=PIPE
    )
    
    process.stdin.write(read_file_bytes('contextlines.yaml'))
    process.stdin.close()

    stdout, stderr = process.communicate()   
    assert stdout.decode().strip() == "Monogatari Series: Second Season"    
    assert stderr.decode().strip() == ""