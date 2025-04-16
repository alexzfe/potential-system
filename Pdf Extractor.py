import os
import re
import pandas as pd
import PyPDF2

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text

def extract_table_data(text):
    """Extract data from a tabular PDF format"""
    data = []
    
    # Extract the region from the title first
    current_region = None
    title_match = re.search(r'RESULTADOS DE (.*?) -', text)
    if title_match:
        current_region = title_match.group(1).strip()
        print(f"Found region in title: {current_region}")
    
    # Skip header lines and results title
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        # Skip empty lines
        if not line:
            continue
            
        # Skip header/title lines
        if "RESULTADOS DE" in line and "EVALUACIÓN PARA EL SERUMS" in line:
            continue
            
        # Skip the header line
        if "APELLIDOS Y NOMBRES" in line and ("PROFESIÓN" in line or "PROFESION" in line):
            continue
            
        # Skip page numbers at the bottom
        if re.match(r'^\d+/\d+$', line):
            continue
            
        lines.append(line)
    
    # Print a sample of lines for debugging
    print(f"Sample of lines to process (first 3):")
    for i in range(min(3, len(lines))):
        print(f"  Line {i+1}: {lines[i]}")
    
    # Complete list of regions for reference
    all_regions = [
        "MADRE DE DIOS", "APURIMAC", "CUSCO", "HUANCAVELICA", "PASCO", 
        "TUMBES", "UCAYALI", "LIMA ESTE", "LIMA SUR", "LIMA CENTRO", "LIMA REGION", 
        "LORETO", "SAN MARTIN", "SAN MARTÍN", "TACNA", "CALLAO", "DIRIS LIMA SUR", 
        "AREQUIPA", "ANCASH", "AMAZONAS", "MOQUEGUA", "HUANUCO", "CAJAMARCA", "PIURA"
        "AYACUCHO", "JUNIN", "LA LIBERTAD", "LAMBAYEQUE", "DIRIS LIMA NORTE", "LIMA NORTE", "PUNO", 'ICA'
    ]
    
    # Complete list of professions for reference
    all_professions = [
        "BIOLOGIA", "ENFERMERIA", "MEDICINA", "MEDICINA VETERINARIA", 
        "NUTRICION", "OBSTETRICIA", "ODONTOLOGIA", "PSICOLOGIA", 
        "QUIMICO FARMACEUTICO", "TM - RADIOLOGIA", "TRABAJO SOCIAL"
    ]
    
    # Process each line
    for line in lines:
        # Check for lines with no space after rank number (like "1CARITA COHAILA...")
        rank_match = re.match(r'^(\d+)([A-ZÑÁÉÍÓÚÜ])', line)
        if rank_match:
            # Fix the line by adding a space after the rank
            rank = rank_match.group(1)
            rest_of_line = rank_match.group(2) + line[len(rank) + 1:]
            print(f"Fixed line format: {rank} {rest_of_line[:50]}...")
        else:
            # Check for normal lines with space after rank
            rank_match = re.match(r'^(\d+)\s+(.*)$', line)
            if rank_match:
                rank = rank_match.group(1)
                rest_of_line = rank_match.group(2)
            else:
                continue  # Not a data line, skip it
        
        # For Tacna and San Martín files, we've seen that the format is:
        # [RANK][NAME] [PROFESSION] [REGION] [SCORE]
        # Find profession and region in the line
        found_profession = None
        for profession in all_professions:
            if profession in rest_of_line:
                found_profession = profession
                break
        
        found_region = None
        for region in all_regions:
            if region in rest_of_line:
                found_region = region
                break
        
        # If we didn't find the region but we know it from the title
        if not found_region and current_region:
            found_region = current_region
        
        # If we found both profession and region
        if found_profession and found_region:
            # Split the line at the profession
            name_parts = rest_of_line.split(found_profession)[0].strip()
            
            # Split the remaining text at the region to get the score
            after_profession = rest_of_line.split(found_profession)[1].strip()
            if found_region in after_profession:
                after_region = after_profession.split(found_region)[1].strip()
                # The score should be at the end
                score_match = re.search(r'(\d+\.\d+|\-)', after_region)
                score = score_match.group(1) if score_match else "-"
            else:
                # If we can't find the region in the string (might happen if region from title),
                # look for a number pattern at the end for score
                score_match = re.search(r'(\d+\.\d+|\-)\s*$', after_profession)
                score = score_match.group(1) if score_match else "-"
            
            # Now create the data entry
            data.append({
                'N°': rank,
                'APELLIDOS Y NOMBRES': name_parts,
                'PROFESIÓN': found_profession,
                'REGIÓN DE EVALUACIÓN': found_region,
                'NOTA': score
            })
            print(f"Extracted: {rank}, {name_parts}, {found_profession}, {found_region}, {score}")
    
    return data

def process_pdfs_in_folder(folder_path):
    """Process all PDFs in the specified folder and return combined data"""
    all_data = []
    
    # Get all PDF files in the folder
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"Processing {pdf_file}...")
        
        try:
            # Extract text from PDF
            text = extract_text_from_pdf(pdf_path)
            
            # Extract data
            data = extract_table_data(text)
            
            # Add to the combined data
            all_data.extend(data)
            
            print(f"  Extracted {len(data)} entries")
        except Exception as e:
            print(f"  Error processing {pdf_file}: {e}")
    
    return all_data

def main():
    # Folder containing the PDFs
    folder_path = r"C:\Users\Alex\Desktop\Analytics\Sewrums"
    
    # Process all PDFs in the folder
    all_data = process_pdfs_in_folder(folder_path)
    
    # Create a DataFrame
    df = pd.DataFrame(all_data)
    
    # Save to CSV with error handling
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(folder_path, f"serums_results_{timestamp}.csv")
    
    try:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nProcessing complete. Combined {len(all_data)} entries into CSV file.")
        print(f"Output saved to: {output_path}")
        if not df.empty:
            print(f"First few rows of the data:")
            print(df.head())
    except Exception as e:
        # Try to save to the Desktop
        try:
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            desktop_output_path = os.path.join(desktop_path, f"serums_results_{timestamp}.csv")
            df.to_csv(desktop_output_path, index=False, encoding='utf-8-sig')
            print(f"Output saved to: {desktop_output_path}")
        except Exception as e2:
            print(f"Could not save file: {e2}")

if __name__ == "__main__":
    main()