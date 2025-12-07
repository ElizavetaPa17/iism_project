FILE_PREFIX = '/home/polina/BSUIR/3rd_grade'

def construct_signature_path(person: str, is_genuine: bool=True, is_surname: bool=False) -> str:
    if not person in persons or not is_genuine and not person in forged_exists:
        raise ValueError(f'Данные для {person}, {is_genuine} не существуют')
    
    template = f'{FILE_PREFIX}/ОЭД/Образцы/{person}/'
    template += 'фамилия' if is_surname else 'подпись' if is_genuine else 'подражание'
    template += '/'
    return template

persons = {'А-1', 'А-2', 'А-3', 'А-4', 'А-5', 'Б-1', 'Б-2', 'Б-3', 'Б-4', 'Б-5', 'Б-6', 
           'В-1', 'В-2', 'В-3', 'В-4', 'В-5', 
           'Г-1', 'Г-2', 'Г-3', 'Г-4', 'Г-5', 'Г-6', 'Г-7', 'Г-8', 'Г-9', 'Г-10', 'Г-11', 'Г-12', 
           'Р-1', 'Р-2',
           'С-1', 'С-2', 'С-3', 'С-4', 'С-5', 'С-6', 'С-7', 'С-8', 'С-9', 'С-10', 'С-11', 'С-12', 
           'Т-1',
           'Ф-2', 'Ф-3', 'Ф-4', 
           'Х-1', 'Х-2', 'Ч-1', 'Ш-1', 'Ш-2', 'Э-1', 'Я-1'}

forged_exists = {'А-1', 'А-5', 'Б-2', 'Б-3', 'Г-2', 'Г-7', 'Г-9', 'С-10'}


TEST_PREFIX = '/home/polina/ПРОЕКТ/Тестовые данные'

test_folders_list = [
    #[f'{TEST_PREFIX}/1/Образцы', f'{TEST_PREFIX}/1'],
    #[f'{TEST_PREFIX}/2/Образцы', f'{TEST_PREFIX}/2'],
    #[f'{TEST_PREFIX}/3/Образцы', f'{TEST_PREFIX}/3'],
    #[f'{TEST_PREFIX}/4/Образцы', f'{TEST_PREFIX}/4'],
    #[f'{TEST_PREFIX}/5/Образцы', f'{TEST_PREFIX}/5'],
    #[f'{TEST_PREFIX}/7/Образцы', f'{TEST_PREFIX}/7'],
    #[f'{TEST_PREFIX}/8/Образцы', f'{TEST_PREFIX}/8'],
    #[f'{TEST_PREFIX}/9/Образцы', f'{TEST_PREFIX}/9'],
    #[f'{TEST_PREFIX}/10/Образцы', f'{TEST_PREFIX}/10'],
    [f'{TEST_PREFIX}/000000000200894/Genuine', f'{TEST_PREFIX}/000000000200894/Forged'],
    [f'{TEST_PREFIX}/000000001045402/Genuine', f'{TEST_PREFIX}/000000001045402/Forged'],
    [f'{TEST_PREFIX}/000000001046474/Genuine', f'{TEST_PREFIX}/000000001046474/Forged'],
    [f'{TEST_PREFIX}/000000001941061/Genuine', f'{TEST_PREFIX}/000000001941061/Forged'],
    [f'{TEST_PREFIX}/000000003150143/Genuine', f'{TEST_PREFIX}/000000003150143/Forged'],
    [f'{TEST_PREFIX}/000000802810034/Genuine', f'{TEST_PREFIX}/000000802810034/Forged'],
    [f'{TEST_PREFIX}/000000814510023/Genuine', f'{TEST_PREFIX}/000000814510023/Forged'],
    [f'{TEST_PREFIX}/000140809082110/Genuine', f'{TEST_PREFIX}/000140809082110/Forged'],
    [f'{TEST_PREFIX}/001810111230005/Genuine', f'{TEST_PREFIX}/001810111230005/Forged'],
    [f'{TEST_PREFIX}/001810111230006/Genuine', f'{TEST_PREFIX}/001810111230006/Forged'],
    [f'{TEST_PREFIX}/002008410100008/Genuine', f'{TEST_PREFIX}/002008410100008/Forged'],
    [f'{TEST_PREFIX}/002008410100011/Genuine', f'{TEST_PREFIX}/002008410100011/Forged'],
    [f'{TEST_PREFIX}/002008410100013/Genuine', f'{TEST_PREFIX}/002008410100013/Forged']
]

'''import os

directory = TEST_PREFIX
test_folders_list = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name)) if len(name) <= 2]

test_folders_list = [
    [f'{TEST_PREFIX}/{folder}/Образцы', f'{TEST_PREFIX}/{folder}']
    for folder in test_folders_list
]'''