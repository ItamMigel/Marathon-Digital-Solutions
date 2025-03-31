import zipfile
from lxml import etree
from datetime import datetime, timedelta

def excel_number_to_datetime(excel_num, date_origin="1900"):
    """
    Конвертирует числовое представление даты из Excel в объект datetime.
    
    Args:
        excel_num: Числовое представление даты в формате Excel
        date_origin: Система отсчета дат в Excel ('1900' или '1904')
        
    Returns:
        datetime: Объект datetime, соответствующий дате Excel
    """
    if date_origin == "1900":
        base_date = datetime(1899, 12, 30)  # Excel считает 1900 високосным (ошибка)
    elif date_origin == "1904":
        base_date = datetime(1904, 1, 1)
    else:
        raise ValueError("Invalid date origin. Use '1900' or '1904'.")

    excel_num = float(excel_num)
    days = int(excel_num)
    fraction = excel_num - days
    total_seconds = fraction * 24 * 60 * 60
    delta = timedelta(seconds=total_seconds)
    
    return base_date + timedelta(days=days) + delta



def get_last_row_fast_xlsx(file_path):
    """
    Быстро извлекает последнюю строку данных из XLSX файла без загрузки всего файла.
    
    Args:
        file_path: Путь к XLSX файлу
        
    Returns:
        list: Список значений из последней строки или None, если файл пуст
    """
    ns = {'ss': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    
    with zipfile.ZipFile(file_path) as z:
        # Читаем sharedStrings.xml (если нужно)
        shared_strings = []
        try:
            with z.open('xl/sharedStrings.xml') as f:
                shared_xml = etree.parse(f)
                shared_strings = [n.text for n in shared_xml.findall('.//ss:t', namespaces=ns)]
        except KeyError:
            pass
        
        with z.open('xl/worksheets/sheet1.xml') as f:
            context = etree.iterparse(f, events=('end',), tag='{*}row')
            last_row = None
            for event, elem in context:
                if last_row is not None:
                    last_row.clear()
                last_row = elem
            
            last_row_values = []

            if last_row is not None:
                for c in last_row.findall('ss:c', namespaces=ns):
                    v = c.find('ss:v', namespaces=ns)
                    cell_value = v.text if v is not None else None
                    
                    # Обработка shared strings
                    if c.get('t') == 's' and cell_value is not None:
                        cell_value = shared_strings[int(cell_value)]
                    
                    # Конвертация даты
                    if c.get('s') == '1':  # Если стиль ячейки соответствует дате
                        try:
                            cell_value = excel_number_to_datetime(float(cell_value))
                        except (ValueError, TypeError):
                            pass
                    
                    last_row_values.append(cell_value)
            return last_row_values
    return None



def get_first_row(file_path):
    """
    Извлекает первую строку из XLSX файла, обрабатывая различные типы данных.
    
    Args:
        file_path: Путь к XLSX файлу
        
    Returns:
        list: Список значений из первой строки или пустой список, если файл пуст
    """
    ns = {'ss': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    
    with zipfile.ZipFile(file_path) as z:
        with z.open('xl/worksheets/sheet1.xml') as f:
            for _, elem in etree.iterparse(f, events=('end',), tag='{*}row'):
                row_data = []
                for c in elem.findall('ss:c', namespaces=ns):
                    # 1. Проверяем inlineStr (встроенные строки)
                    if c.get('t') == 'inlineStr':
                        is_elem = c.find('ss:is', namespaces=ns)
                        if is_elem is not None:
                            t_elem = is_elem.find('ss:t', namespaces=ns)
                            if t_elem is not None:
                                row_data.append(t_elem.text or '')
                                continue
                    
                    # 2. Проверяем shared strings
                    if c.get('t') == 's':
                        try:
                            with z.open('xl/sharedStrings.xml') as ss_file:
                                shared_xml = etree.parse(ss_file)
                                shared_strings = [t.text or '' for t in shared_xml.findall('.//ss:t', namespaces=ns)]
                                v = c.find('ss:v', namespaces=ns)
                                if v is not None and v.text is not None:
                                    idx = int(v.text)
                                    row_data.append(shared_strings[idx] if idx < len(shared_strings) else '')
                                    continue
                        except (KeyError, ValueError, IndexError):
                            pass
                
                elem.clear()
                return row_data  # Возвращаем первую найденную строку
    
    return []  # Если файл пустой


def get_last_info(path):
    """
    Получает последние измерения из XLSX файла, преобразуя их в словарь.
    
    Args:
        path: Путь к XLSX файлу с данными измерений
        
    Returns:
        dict: Словарь, где ключи - названия параметров из первой строки,
              а значения - соответствующие значения из последней строки
    """
    last_row = get_last_row_fast_xlsx(path)
    first_row = get_first_row(path)

    # Преобразование типов данных
    for i in range(len(last_row)):
        if first_row[i] == 'Время':
            if isinstance(last_row[i], str):
                last_row[i] = excel_number_to_datetime(last_row[i])
        else:
            last_row[i] = float(last_row[i])

    # Создание словаря с данными
    info = {first_row[i]: last_row[i] for i in range(len(last_row))}
    
    return info


if __name__ == '__main__':
    # Тестовый запуск для проверки функциональности
    info = get_last_info(path='backend/test_data.xlsx')
    print(info)