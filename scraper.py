import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from bs4 import BeautifulSoup
import pandas as pd
import time

# --- Your Comprehensive URL List ---
URL_LIST = [
    "https://www.lamudi.co.id/jual/banten/tangerang/gading-serpong/rumah/",
    "https://www.lamudi.co.id/jual/banten/tangerang/cikupa/rumah/",
    "https://www.lamudi.co.id/jual/banten/tangerang/cipondoh/rumah/",
    "https://www.lamudi.co.id/jual/banten/tangerang/karawaci/rumah/",
    "https://www.lamudi.co.id/jual/banten/tangerang/pagedangan/rumah/",
    "https://www.lamudi.co.id/jual/banten/tangerang/cisauk/rumah/",
    "https://www.lamudi.co.id/jual/banten/tangerang/tangerang-1/rumah/",
    "https://www.lamudi.co.id/jual/banten/tangerang/cipayung/rumah/",
    "https://www.lamudi.co.id/jual/banten/tangerang/sukajadi/rumah/",
    "https://www.lamudi.co.id/jual/jakarta/jakarta-selatan/rumah/",
    "https://www.lamudi.co.id/jual/jakarta/jakarta-barat/rumah/",
    "https://www.lamudi.co.id/jual/jakarta/jakarta-timur/rumah/",
    "https://www.lamudi.co.id/jual/jakarta/jakarta-pusat/rumah/",
    "https://www.lamudi.co.id/jual/jakarta/jakarta-utara/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/depok/sawangan-1/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/depok/cilodong/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/depok/cimanggis/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/depok/cinere/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/depok/pancoran-mas/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/depok/sukmajaya/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/depok/citayam/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/depok/cipayung-2/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/depok/beji-3/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/depok/limo/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/bekasi/harapan-indah/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/bekasi/bekasi-utara/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/bekasi/medan-satria/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/bekasi/jatiasih/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/bekasi/babelan/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/bekasi/bekasi-barat/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/bekasi/pondok-gede/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/bekasi/jatisampurna/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/bekasi/bekasi-selatan/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/bekasi/tarumajaya/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/bogor/cibinong/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/bogor/cileungsi/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/bogor/jonggol/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/bogor/gunung-putri/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/bogor/sentul-city-1/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/bogor/gunung-sindur/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/bogor/parung/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/bogor/tanah-sareal/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/bogor/bogor-barat---kota/rumah/",
    "https://www.lamudi.co.id/jual/jawa-barat/bogor/bojonggede/rumah/"
]

def scrape_live_data():
    options = uc.ChromeOptions()
    driver = uc.Chrome(options=options, use_subprocess=True)
    
    all_properties = []
    
    try:
        print(f"Opening first page for security check: {URL_LIST[0]}")
        driver.get(URL_LIST[0])
        driver.maximize_window()
        input("\n>>> Please solve any CAPTCHA in the browser window. \n>>> Once you see the property listings, press Enter here to begin...")

        for base_url in URL_LIST:
            print(f"\n--- Starting new area: {base_url} ---")
            driver.get(base_url)
            page_count = 1
            
            while True:
                try:
                    print(f"Processing page {page_count}...")
                    WebDriverWait(driver, 20).until(
                        EC.visibility_of_element_located((By.CLASS_NAME, "ListingCellItem_cellItemWrapper__t2hO2"))
                    )
                    
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    property_cards = soup.find_all('div', class_='ListingCellItem_cellItemWrapper__t2hO2')
                    
                    if not property_cards:
                        print("No listings found on this page, but continuing.")
                    
                    print(f"Found {len(property_cards)} properties on this page. Extracting data...")
                    for card in property_cards:
                        property_data = {}
                        try:
                            property_data['Title'] = card.find('h3', class_='ListingCellItem_listingTitle__lHzmY').text
                            property_data['Address'] = card.find('span', class_='ListingCellItem_addressLine__hp5ZO').text
                            property_data['Price'] = card.find('span', class_='ListingCellItem_listingPrice___oTdU').text
                            
                            bedrooms, building_area, land_area = None, None, None
                            attributes = card.find_all('div', class_='ListingCellItem_attributeItem__d9TFw')
                            for attr in attributes:
                                value = attr.text
                                icon_element = attr.find('span')
                                if icon_element:
                                    icon_class = icon_element.get('class', [])
                                    if 'icon-bedrooms' in icon_class: bedrooms = value
                                    elif 'icon-livingsize' in icon_class: building_area = value
                                    elif 'icon-land_size' in icon_class: land_area = value
                            
                            property_data['Bedrooms'] = bedrooms
                            property_data['Building Area (m²)'] = building_area
                            property_data['Land Area (m²)'] = land_area
                            
                            all_properties.append(property_data)
                        except Exception:
                            continue

                    next_button_found_and_active = False
                    try:
                        pagination = driver.find_element(By.CLASS_NAME, "Pagination_pagination__cRFTg")
                        nav_links = pagination.find_elements(By.TAG_NAME, "a")
                        for link in nav_links:
                            if link.find_elements(By.CLASS_NAME, 'icon-play-right'):
                                if "Pagination_disabled__J4eEO" not in link.get_attribute("class"):
                                    driver.execute_script("arguments[0].click();", link)
                                    next_button_found_and_active = True
                                break
                    except NoSuchElementException:
                        pass
                    
                    if not next_button_found_and_active:
                        print("Reached the last page for this area.")
                        break
                    
                    page_count += 1
                    time.sleep(4)

                except TimeoutException:
                    print(f"Timed out waiting for listings. Moving to next area.")
                    break
        
        # --- Save the clean data to a CSV file ---
        if all_properties:
            df = pd.DataFrame(all_properties)
            # Reorder columns to make Title the first column
            df = df[['Title', 'Address', 'Price', 'Bedrooms', 'Building Area (m²)', 'Land Area (m²)']]
            df.to_csv('properti_data.csv', index=False, encoding='utf-8')
            print(f"\n✅ Success! Saved {len(df)} properties to 'properti_data.csv'.")
            print("\n--- Preview of Your Data ---")
            print(df.head())

    except Exception as e:
        print(f"\n❌ A critical error occurred: {e}")

    finally:
        driver.quit()

if __name__ == "__main__":
    scrape_live_data()