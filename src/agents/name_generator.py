"""Name generation with variety based on demographics and culture."""
import random
from typing import List, Optional, Tuple


class NameGenerator:
    """Generates diverse names based on cultural and demographic factors."""
    
    # Common first names by cultural background and gender
    FIRST_NAMES = {
        'western': {
            'neutral': ['Alex', 'Jordan', 'Taylor', 'Morgan', 'Casey', 'Riley', 'Avery', 'Quinn'],
            'traditional': ['James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 
                          'Michael', 'Linda', 'William', 'Elizabeth', 'David', 'Barbara',
                          'Richard', 'Susan', 'Joseph', 'Jessica', 'Thomas', 'Sarah'],
            'modern': ['Aiden', 'Emma', 'Liam', 'Olivia', 'Noah', 'Ava', 'Mason', 'Isabella',
                      'Ethan', 'Sophia', 'Logan', 'Mia', 'Lucas', 'Charlotte', 'Jackson', 'Amelia']
        },
        'hispanic': {
            'neutral': ['Angel', 'Guadalupe', 'Alexis', 'Andrea'],
            'traditional': ['Juan', 'Maria', 'Jose', 'Carmen', 'Luis', 'Rosa', 'Carlos', 'Ana',
                          'Jorge', 'Isabel', 'Pedro', 'Lucia', 'Miguel', 'Elena', 'Antonio', 'Sofia'],
            'modern': ['Santiago', 'Valentina', 'Mateo', 'Camila', 'Sebastian', 'Victoria',
                     'Diego', 'Natalia', 'Adrian', 'Luna', 'Daniel', 'Mariana']
        },
        'asian': {
            'neutral': ['Min', 'Wei', 'Lin', 'Yu', 'Chen', 'Kim'],
            'traditional': ['Hiroshi', 'Yuki', 'Takashi', 'Akiko', 'Kenji', 'Mei', 
                          'Jun', 'Ling', 'Ravi', 'Priya', 'Raj', 'Anita', 'Amit', 'Kavya'],
            'modern': ['Ken', 'Amy', 'Steve', 'Grace', 'Kevin', 'Michelle', 'Brian', 'Jessica',
                     'Eric', 'Jennifer', 'Alan', 'Lisa', 'Ryan', 'Emily']
        },
        'african': {
            'neutral': ['Amari', 'Zuri', 'Asha', 'Jahi', 'Kali', 'Nile'],
            'traditional': ['Kwame', 'Amara', 'Jabari', 'Nia', 'Malik', 'Zara', 'Darius', 'Aisha',
                          'Omar', 'Fatima', 'Ibrahim', 'Khadija', 'Yusuf', 'Layla'],
            'modern': ['DeShawn', 'Jasmine', 'Tyrone', 'Aaliyah', 'Jamal', 'Destiny',
                     'Marcus', 'Diamond', 'Andre', 'Ebony', 'Terrell', 'Jade']
        },
        'middle_eastern': {
            'neutral': ['Noor', 'Sam', 'Rami', 'Dana', 'Zain', 'Iman'],
            'traditional': ['Ahmed', 'Fatima', 'Mohammed', 'Aisha', 'Ali', 'Zahra', 
                          'Hassan', 'Maryam', 'Omar', 'Leila', 'Khalid', 'Sara'],
            'modern': ['Adam', 'Maya', 'Zane', 'Lara', 'Karim', 'Yasmin', 'Tariq', 'Nadia']
        },
        'european': {
            'neutral': ['Alex', 'Charlie', 'Max', 'Sam', 'Adrian', 'Robin'],
            'traditional': ['Pierre', 'Marie', 'Hans', 'Greta', 'Giovanni', 'Maria',
                          'Ivan', 'Olga', 'Sven', 'Ingrid', 'Klaus', 'Heidi'],
            'modern': ['Luca', 'Emma', 'Felix', 'Sophie', 'Leon', 'Mia', 'Finn', 'Ella',
                     'Oscar', 'Clara', 'Hugo', 'Luna', 'Arthur', 'Alice']
        }
    }
    
    # Common last names by cultural background
    LAST_NAMES = {
        'western': ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Miller', 'Davis', 
                   'Garcia', 'Rodriguez', 'Wilson', 'Martinez', 'Anderson', 'Taylor',
                   'Thomas', 'Hernandez', 'Moore', 'Martin', 'Jackson', 'Thompson', 'White',
                   'Lopez', 'Lee', 'Gonzalez', 'Harris', 'Clark', 'Lewis', 'Robinson',
                   'Walker', 'Perez', 'Hall', 'Young', 'Allen', 'King', 'Wright', 'Scott'],
        'hispanic': ['Garcia', 'Rodriguez', 'Martinez', 'Lopez', 'Gonzalez', 'Hernandez',
                    'Perez', 'Sanchez', 'Ramirez', 'Torres', 'Rivera', 'Gomez', 'Diaz',
                    'Cruz', 'Reyes', 'Morales', 'Ortiz', 'Gutierrez', 'Chavez', 'Ramos',
                    'Vargas', 'Castillo', 'Jimenez', 'Moreno', 'Romero', 'Herrera'],
        'asian': ['Wang', 'Li', 'Zhang', 'Liu', 'Chen', 'Yang', 'Huang', 'Zhao', 'Wu',
                 'Kim', 'Park', 'Lee', 'Choi', 'Jung', 'Kang', 'Yamamoto', 'Tanaka',
                 'Sato', 'Suzuki', 'Takahashi', 'Patel', 'Singh', 'Kumar', 'Sharma'],
        'african': ['Okafor', 'Nkomo', 'Mbeki', 'Diallo', 'Kamara', 'Toure', 'Mensah',
                   'Asante', 'Okello', 'Mwangi', 'Ndlovu', 'Dube', 'Moyo', 'Banda'],
        'middle_eastern': ['Al-Rahman', 'Hassan', 'Abdullah', 'Al-Salem', 'Malik',
                          'Rashid', 'Nasser', 'Khalil', 'Habib', 'Farah', 'Haddad',
                          'Khoury', 'Najjar', 'Sabbagh', 'Shadid'],
        'european': ['Mueller', 'Schmidt', 'Schneider', 'Fischer', 'Weber', 'Meyer',
                    'Rossi', 'Russo', 'Ferrari', 'Esposito', 'Bianchi', 'Romano',
                    'Dubois', 'Moreau', 'Laurent', 'Simon', 'Michel', 'Leroy',
                    'Novak', 'Dvorak', 'Petrov', 'Ivanov', 'Popov', 'Sokolov']
    }
    
    @classmethod
    def generate_name(cls, cultural_tags: List[str], age: int, 
                     socioeconomic_tags: List[str]) -> Tuple[str, str]:
        """Generate a culturally appropriate name.
        
        Returns:
            Tuple of (first_name, last_name)
        """
        # Determine cultural background
        culture = cls._determine_culture(cultural_tags)
        
        # Determine name style based on age and other factors
        if age > 50 or 'traditional' in cultural_tags:
            style = 'traditional'
        else:
            style = 'modern'
        
        # Get appropriate name pools
        first_names_pool = cls.FIRST_NAMES[culture].get(style, [])
        if not first_names_pool:
            first_names_pool = cls.FIRST_NAMES[culture]['neutral']
        
        # Add more variety by sometimes using neutral names
        if random.random() < 0.2:
            first_names_pool.extend(cls.FIRST_NAMES[culture]['neutral'])
        
        last_names_pool = cls.LAST_NAMES[culture]
        
        # Select names
        first_name = random.choice(first_names_pool)
        last_name = random.choice(last_names_pool)
        
        return first_name, last_name
    
    @classmethod
    def _determine_culture(cls, cultural_tags: List[str]) -> str:
        """Determine cultural background from tags."""
        # Check for explicit cultural indicators
        culture_mapping = {
            'western': ['american', 'british', 'canadian', 'australian'],
            'hispanic': ['hispanic', 'latino', 'mexican', 'spanish', 'latin'],
            'asian': ['asian', 'chinese', 'japanese', 'korean', 'indian', 'vietnamese'],
            'african': ['african', 'nigerian', 'kenyan', 'ghanaian', 'ethiopian'],
            'middle_eastern': ['middle-eastern', 'arab', 'persian', 'turkish'],
            'european': ['european', 'french', 'german', 'italian', 'russian', 'polish']
        }
        
        for culture, indicators in culture_mapping.items():
            if any(indicator in ' '.join(cultural_tags).lower() for indicator in indicators):
                return culture
        
        # Default based on other tags
        if 'progressive' in cultural_tags or 'secular' in cultural_tags:
            return random.choice(['western', 'european'])
        elif 'traditional' in cultural_tags:
            return random.choice(['hispanic', 'asian', 'middle_eastern'])
        else:
            # Random selection with weighted distribution
            cultures = list(cls.FIRST_NAMES.keys())
            weights = [0.35, 0.20, 0.20, 0.10, 0.10, 0.05]  # Favor western/hispanic
            return random.choices(cultures, weights=weights)[0]
    
    @classmethod
    def format_full_name(cls, first_name: str, last_name: str, 
                        cultural_tags: List[str]) -> str:
        """Format the full name appropriately."""
        # Some cultures use different name orders
        if any(tag in ' '.join(cultural_tags).lower() for tag in ['asian', 'chinese', 'korean', 'vietnamese']):
            # Sometimes use Asian name order
            if random.random() < 0.3:
                return f"{last_name} {first_name}"
        
        return f"{first_name} {last_name}"