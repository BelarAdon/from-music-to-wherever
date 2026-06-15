outfit_mapping = {

    # ══════════════════════════════════════════════════════
    # CLUSTER 0 — CONCIERTO / EUFORIA
    # Base: 3 prendas (top + pantalón + chaqueta)
    # Estilo reemplaza las primeras N prendas del base
    # ══════════════════════════════════════════════════════
    0: {
        "mood_name": "Concierto / Euforia",

        "outfit_base": {
            "prendas": ["top de lentejuelas o mesh", "pantalón vinilo o cuero sintético", "chaqueta con tachuelas"],
            "accesorios": ["gafas con cristales de colores", "pendientes XL statement"],
            "justificacion": "Prendas con brillo y textura que gritan energía desde el primer vistazo."
        },

        "por_estilo": {
            # Reemplaza: top → corset, pantalón → pitillo vinilo, chaqueta → igual base
            "femenino": {
                "prendas": ["corset externo brillante", "pantalón pitillo de vinilo"],
                "accesorios": ["botas hasta la rodilla con plataforma"],
                "justificacion": "El corset como pieza estrella lleva el look al siguiente nivel."
            },
            # Reemplaza: top → camiseta tour, pantalón → cargo reflectante, chaqueta → igual base
            "masculino": {
                "prendas": ["camiseta gráfica de banda o tour", "pantalón cargo reflectante"],
                "accesorios": ["cadenas plateadas en capas"],
                "justificacion": "Referencia directa a la cultura de conciertos con actitud rock urbana."
            },
            # Reemplaza: top → camiseta festival, pantalón → igual base, chaqueta → bomber
            "unisex": {
                "prendas": ["camiseta oversized de festival", "pantalón cargo oscuro", "bomber con parches o bordados"],
                "accesorios": ["riñonera metálica cruzada"],
                "justificacion": "La bomber con parches es el uniforme no oficial de los festivales."
            },
            # Reemplaza todo: hoodie + cargo + chaqueta técnica
            "streetwear": {
                "prendas": ["hoodie técnico cortaviento oversize", "pantalón cargo reflectante", "chaqueta técnica de running"],
                "accesorios": ["gorra snapback"],
                "justificacion": "Streetwear de alto impacto visual, funcional para moverse entre la multitud."
            },
            # Reemplaza todo: el minimal aquí es negro total bien cortado
            "minimal": {
                "prendas": ["top negro sin mangas escote recto", "pantalón cigarette negro de talle alto", "blazer negro oversized"],
                "accesorios": ["ear cuff dorado fino"],
                "justificacion": "Negro total de arriba abajo: el minimalismo también puede ser poderoso."
            },
            # Reemplaza todo: cuero + asimétrico + tachuelas
            "edgy": {
                "prendas": ["top asimétrico negro", "pantalón de cuero con cremalleras laterales", "chaqueta con tachuelas y hombreras"],
                "accesorios": ["collar de pinchos o cadena gruesa"],
                "justificacion": "Estética punk-rock total que encaja a la perfección con la energía del concierto."
            },
        },

        "por_estacion": {
            "invierno": {
                "prendas": ["abrigo de pelo sintético en color llamativo"],
                "accesorios": ["guantes de cuero fino"],
                "justificacion": "El abrigo de pelo mantiene el impacto visual incluso en la cola del concierto."
            },
            "verano": {
                "prendas": ["shorts de tiro alto metalizados"],
                "accesorios": ["gafas de sol wraparound"],
                "justificacion": "Máxima expresión del festival de verano: brillo, piel y sol."
            },
            "primavera": {
                "prendas": ["chaqueta denim customizada con parches"],
                "accesorios": ["pañuelo en el cuello"],
                "justificacion": "La denim jacket customizada es icónica en conciertos de primavera."
            },
            "otoño": {
                "prendas": ["chaqueta bomber de satin"],
                "accesorios": ["botines con puntera metálica"],
                "justificacion": "El satén de la bomber capta la luz artificial del escenario en otoño."
            },
        },

        "por_clima": {
            "lluvia": {
                "prendas": ["impermeable transparente de PVC"],
                "accesorios": ["botas de agua negras con plataforma"],
                "justificacion": "El PVC transparente protege sin tapar el outfit — icónico en festivales."
            },
            "frio": {
                "prendas": ["forro polar técnico oversize"],
                "accesorios": ["gorro beanie con logo"],
                "justificacion": "Abrigo funcional sin perder la actitud del concierto."
            },
            "calor": {
                "prendas": ["top bandeau o crop sin mangas"],
                "accesorios": ["abanico de mano decorativo"],
                "justificacion": "Mínima ropa, máximo impacto en clima caluroso."
            },
        }
    },

    # ══════════════════════════════════════════════════════
    # CLUSTER 1 — INTENSO / DRAMÁTICO
    # Base: camisa satén + pantalón estructurado + gabardina
    # ══════════════════════════════════════════════════════
    1: {
        "mood_name": "Intenso / Dramático",

        "outfit_base": {
            "prendas": ["camisa negra de satén o seda", "pantalón de talle alto estructurado oscuro", "gabardina larga"],
            "accesorios": ["anillos statement en varios dedos"],
            "justificacion": "Silueta larga y oscura que impone presencia sin necesitar adornos."
        },

        "por_estilo": {
            # Reemplaza: camisa → blusa asimétrica, pantalón → falda midi, gabardina queda
            "femenino": {
                "prendas": ["blusa de satén con escote cruzado", "falda midi de talle alto asimétrica", "gabardina cruzada larga"],
                "accesorios": ["pendientes colgantes geométricos de metal"],
                "justificacion": "La asimetría añade tensión visual que intensifica el dramatismo."
            },
            # Reemplaza: camisa → chaqueta sastre, pantalón queda, gabardina queda
            "masculino": {
                "prendas": ["chaqueta sastre oversized en tono oscuro", "pantalón de talle alto estructurado oscuro", "gabardina larga"],
                "accesorios": ["cinturón de cuero grueso"],
                "justificacion": "La americana oversized da poder y estructura sin esfuerzo."
            },
            # Reemplaza: camisa → top liso, pantalón → palazzo, gabardina → kimono largo
            "unisex": {
                "prendas": ["top de punto liso cuello redondo", "pantalón palazzo negro", "kimono largo negro"],
                "accesorios": ["pulsera de cuero trenzada"],
                "justificacion": "El kimono largo aporta movimiento y drama en cualquier género."
            },
            # Reemplaza todo por look urbano oscuro
            "streetwear": {
                "prendas": ["sudadera negra con capucha y cremallera completa", "pantalón cargo negro técnico", "chaqueta multipocket oscura"],
                "accesorios": ["riñonera táctica negra"],
                "justificacion": "Streetwear oscuro con referencias militares y urbanas muy intensas."
            },
            # Reemplaza todo: negro puro de corte limpio
            "minimal": {
                "prendas": ["top de tirantes con escote en V profundo negro", "pantalón palazzo negro de talle alto", "abrigo recto negro largo"],
                "accesorios": ["anillo solitario de plata grande"],
                "justificacion": "Menos prendas, más impacto: el negro puro como declaración estética."
            },
            # Reemplaza todo: cuero y elementos de fuerza
            "edgy": {
                "prendas": ["camiseta negra de tirantes bajo chaqueta de cuero con hombreras", "pantalón pitillo negro rasgado", "chaqueta de cuero con hombreras"],
                "accesorios": ["collar de pinchos doble vuelta"],
                "justificacion": "La combinación de cuero y hombreras lleva el dramatismo a su versión más extrema."
            },
        },

        "por_estacion": {
            "invierno": {
                "prendas": ["abrigo largo de lana en negro o burdeos"],
                "accesorios": ["guantes de cuero negro"],
                "justificacion": "El abrigo largo oscuro en invierno es la pieza dramática por excelencia."
            },
            "verano": {
                "prendas": ["camiseta sin mangas de lino negro"],
                "accesorios": ["gafas de sol negras rectangulares o cat-eye"],
                "justificacion": "El lino negro es fresco y mantiene intacta la intensidad del mood."
            },
            "primavera": {
                "prendas": ["trench coat oscuro de corte cruzado"],
                "accesorios": ["botas Chelsea negras de cuero"],
                "justificacion": "El trench cruzado tiene algo cinematográfico perfecto para primavera."
            },
            "otoño": {
                "prendas": ["chaqueta de terciopelo en burdeos o negro"],
                "accesorios": ["botines de tacón cuadrado"],
                "justificacion": "El terciopelo en otoño intensifica el dramatismo con textura y profundidad."
            },
        },

        "por_clima": {
            "lluvia": {
                "prendas": ["gabardina impermeable de corte cruzado"],
                "accesorios": ["botas de agua negras hasta la rodilla"],
                "justificacion": "La gabardina cruzada es funcional y mantiene la silueta dramática intacta."
            },
            "frio": {
                "prendas": ["jersey de cuello alto de canalé en negro"],
                "accesorios": ["guantes sin dedos de cuero negro"],
                "justificacion": "El cuello alto negro es la prenda más dramática del invierno."
            },
            "calor": {
                "prendas": ["camiseta de tirantes de seda o satén negro"],
                "accesorios": ["gafas de sol espejadas en negro o plata"],
                "justificacion": "La seda negra es fresca, fluida y mantiene la elegancia oscura."
            },
        }
    },

    # ══════════════════════════════════════════════════════
    # CLUSTER 2 — INSTRUMENTAL / CONCENTRACIÓN
    # Base: pantalón chino + camisa lino — tonos tierra y neutros
    # ══════════════════════════════════════════════════════
    2: {
        "mood_name": "Instrumental / Concentración",

        "outfit_base": {
            "prendas": ["pantalón chino de algodón en beige o verde oliva", "camisa de lino manga larga"],
            "accesorios": ["tote bag de lona natural"],
            "justificacion": "Tonos tierra y tejidos naturales que transmiten calma y claridad mental."
        },

        "por_estilo": {
            # Reemplaza: pantalón → falda midi, camisa → blusa suave
            "femenino": {
                "prendas": ["falda midi plisada en tono neutro (arena o salvia)", "blusa suelta de algodón o lino"],
                "accesorios": ["pendientes de aro fino dorado"],
                "justificacion": "La falda plisada fluye con el movimiento y aporta elegancia sin esfuerzo."
            },
            # Reemplaza: pantalón → técnico slim, camisa → sudadera técnica
            "masculino": {
                "prendas": ["pantalón chino slim en gris o azul pizarra", "sudadera técnica de cuello con cremallera corta"],
                "accesorios": ["reloj de esfera minimalista"],
                "justificacion": "Lo técnico slim da un look ordenado y productivo."
            },
            # Reemplaza todo: mono como pieza única
            "unisex": {
                "prendas": ["mono de trabajo oversize en beige o crema", "camiseta interior blanca visible"],
                "accesorios": ["calcetines gruesos de lana visibles sobre deportivas"],
                "justificacion": "El mono es una sola pieza con todo resuelto: máxima concentración."
            },
            # Reemplaza: pantalón → técnico jogger, camisa → manga larga térmica
            "streetwear": {
                "prendas": ["pantalón técnico jogger en gris antracita", "camiseta de manga larga térmica en blanco o gris"],
                "accesorios": ["zapatillas de running en tono neutro"],
                "justificacion": "Streetwear funcional orientado al movimiento sin distracciones visuales."
            },
            # Reemplaza: pantalón → recto crema, camisa → canalé manga larga
            "minimal": {
                "prendas": ["pantalón recto de talle alto en crema o marfil", "camiseta de canalé manga larga en el mismo tono"],
                "accesorios": ["gafas de pasta fina en tono carey o negro"],
                "justificacion": "Tono sobre tono en crema o marfil: la paleta visual de la concentración."
            },
            # Reemplaza todo: negro técnico total
            "edgy": {
                "prendas": ["pantalón negro slim técnico con bolsillos laterales", "camiseta negra de cuello redondo slim de algodón pesado"],
                "accesorios": ["reloj negro mate de correa de nylon"],
                "justificacion": "El negro total minimalista-técnico tiene su propio tipo de intensidad."
            },
        },

        "por_estacion": {
            "invierno": {
                "prendas": ["jersey chunky knit en crema, camel o gris claro"],
                "accesorios": ["calcetines de cachemira en tono neutro"],
                "justificacion": "El punto grueso en tono claro es la prenda de concentración invernal por excelencia."
            },
            "verano": {
                "prendas": ["pantalón de lino ancho en blanco o beige", "camiseta de algodón pima de manga corta"],
                "accesorios": ["sandalias de cuero planas con tira fina"],
                "justificacion": "El lino ancho es fresco, holgado y perfecto para los días de trabajo en verano."
            },
            "primavera": {
                "prendas": ["chaqueta harrington en verde oliva o beige"],
                "accesorios": ["bolsa mensajero de lona resistente"],
                "justificacion": "La harrington es la chaqueta ligera más versátil para la primavera."
            },
            "otoño": {
                "prendas": ["cardigan largo de punto en marrón, terracota o burdeos apagado"],
                "accesorios": ["botas de piel marrón con cordones al tobillo"],
                "justificacion": "El cardigan largo en colores otoñales combina concentración con calidez visual."
            },
        },

        "por_clima": {
            "lluvia": {
                "prendas": ["anorak técnico ligero impermeable en color neutro"],
                "accesorios": ["mochila impermeable de roll-top en lona o nylon"],
                "justificacion": "El anorak técnico protege sin añadir capas ni ruido visual."
            },
            "frio": {
                "prendas": ["forro polar de cuello alto en gris o verde musgo"],
                "accesorios": ["gorro de punto fino sin pompón en tono neutro"],
                "justificacion": "El polar de cuello alto es funcional y cómodo para trabajar con frío."
            },
            "calor": {
                "prendas": ["camiseta sin mangas de algodón orgánico en tono neutro"],
                "accesorios": ["abanico plegable de madera o bambú"],
                "justificacion": "Lo más ligero posible sin perder la estética natural del mood."
            },
        }
    },

    # ══════════════════════════════════════════════════════
    # CLUSTER 3 — CHILL GROOVE / URBANO SUAVE
    # Base: jogger de felpa + waffle knit — tonos cálidos neutros
    # ══════════════════════════════════════════════════════
    3: {
        "mood_name": "Chill Groove / Urbano suave",

        "outfit_base": {
            "prendas": ["pantalón jogger de felpa en tono moca o marrón claro", "camiseta de manga larga de tejido waffle"],
            "accesorios": ["zapatillas de cuero blancas o crema"],
            "justificacion": "Texturas suaves y tonos cálidos que transmiten relajación urbana sin descuido."
        },

        "por_estilo": {
            # Reemplaza: jogger → pantalón punto, waffle → crop top de punto a juego
            "femenino": {
                "prendas": ["pantalón de punto acanalado de talle alto en tono neutro", "crop top de punto a juego"],
                "accesorios": ["bolso baguette mini de ante en camel o nude"],
                "justificacion": "El conjunto de punto coordinado es la definición del chill chic."
            },
            # Reemplaza: jogger queda, waffle → camiseta vintage
            "masculino": {
                "prendas": ["pantalón jogger de felpa en moca", "camiseta oversized con lavado vintage en blanco roto"],
                "accesorios": ["gorro bucket de algodón en color neutro"],
                "justificacion": "El lavado vintage aporta historia y relajación sin parecer descuidado."
            },
            # Reemplaza todo: set de terciopelo como pieza única
            "unisex": {
                "prendas": ["pantalón de terciopelo suave en color neutro (gris perla o topo)", "sudadera a juego de terciopelo"],
                "accesorios": ["riñonera de nylon en tono neutro"],
                "justificacion": "El set de terciopelo coordinado es la prenda chill definitiva."
            },
            # Reemplaza: jogger → cargo nylon, waffle → manga larga técnica
            "streetwear": {
                "prendas": ["pantalón cargo de nylon suave en verde o gris", "manga larga técnica con media cremallera"],
                "accesorios": ["gorra de camionero mesh en color neutro"],
                "justificacion": "Streetwear suavizado con tejidos técnicos de tacto agradable."
            },
            # Reemplaza todo: wide leg + supima = chill limpio
            "minimal": {
                "prendas": ["pantalón wide leg en crema o gris claro de talle alto", "camiseta de algodón supima blanca o crema"],
                "accesorios": ["reloj de correa de tela en tono neutro"],
                "justificacion": "El wide leg holgado en tono claro es minimalismo chill en estado puro."
            },
            # Reemplaza: jogger → punto acanalado negro, waffle → manga larga slim negro
            "edgy": {
                "prendas": ["pantalón negro de punto acanalado slim", "camiseta negra de manga larga slim de algodón"],
                "accesorios": ["botas de ante negro con suela gruesa y cremallera lateral"],
                "justificacion": "El chill también puede tener actitud: negro suave con suela potente."
            },
        },

        "por_estacion": {
            "invierno": {
                "prendas": ["puffer coat corto en camel, beis o negro"],
                "accesorios": ["bufanda de punto grueso en color arena o crema"],
                "justificacion": "El puffer corto en neutro es la prenda urbana más chill del invierno."
            },
            "verano": {
                "prendas": ["shorts de felpa fina o terry en beige o blanco roto"],
                "accesorios": ["gafas de sol redondas de montura fina dorada"],
                "justificacion": "Los shorts de felpa son el equivalente veraniego del jogger chill."
            },
            "primavera": {
                "prendas": ["chaqueta de punto abierta en color tierra o salvia"],
                "accesorios": ["pañuelo de seda o algodón anudado al cuello"],
                "justificacion": "La chaqueta de punto abierta es perfecta para días suaves de primavera."
            },
            "otoño": {
                "prendas": ["cardigan oversized en marrón chocolate o terracota"],
                "accesorios": ["gorro de lana en tono topo o gris"],
                "justificacion": "El cardigan oversized en colores otoñales es el chill hecho ropa."
            },
        },

        "por_clima": {
            "lluvia": {
                "prendas": ["chubasquero oversize en color pastel o nude"],
                "accesorios": ["zapatillas impermeables de gore-tex en blanco o gris"],
                "justificacion": "El chubasquero pastel mantiene la vibra suave incluso en día gris."
            },
            "frio": {
                "prendas": ["jersey de punto flojo cuello redondo en camel o arena"],
                "accesorios": ["bufanda infinita de lana merino en tono neutro"],
                "justificacion": "El jersey flojo de camel es la prenda más reconfortante del frío urbano."
            },
            "calor": {
                "prendas": ["camiseta de tirantes acanalada en color mantequilla o melocotón"],
                "accesorios": ["gorra de béisbol de lona lavada en tono neutro"],
                "justificacion": "Tirantes acanalados en tono cálido: chill de verano sin esfuerzo."
            },
        }
    },

    # ══════════════════════════════════════════════════════
    # CLUSTER 4 — HAPPY / BUEN ROLLO
    # Base: pantalón de pinzas color + camisa de bowling
    # ══════════════════════════════════════════════════════
    4: {
        "mood_name": "Happy / Buen Rollo",

        "outfit_base": {
            "prendas": ["pantalón ancho de pinzas en color vivo o pastel", "camisa de bowling estampada o de rayas alegres"],
            "accesorios": ["zapatillas de lona de color o con estampado"],
            "justificacion": "Colores y estampados que comunican alegría desde lejos sin gritar."
        },

        "por_estilo": {
            # Reemplaza: pantalón → vestido camisero (outfit completo), sin camisa
            "femenino": {
                "prendas": ["vestido camisero floral o de lunares midi", "cárdigan fino de color pastel encima"],
                "accesorios": ["diadema de tela o con flores pequeñas"],
                "justificacion": "El vestido camisero estampado con cárdigan pastel encima es el look happy más completo."
            },
            # Reemplaza: pantalón queda, camisa → hawaiana abierta sobre camiseta blanca
            "masculino": {
                "prendas": ["pantalón ancho de pinzas en color neutro o pastel", "camisa hawaiana de resort abierta", "camiseta blanca debajo"],
                "accesorios": ["gafas de sol de colores con montura gruesa"],
                "justificacion": "La hawaiana abierta sobre camiseta blanca es el icono del buen rollo veraniego."
            },
            # Reemplaza: pantalón queda, camisa → tie-dye
            "unisex": {
                "prendas": ["pantalón ancho de pinzas en color vivo", "camiseta tie-dye o de estampado abstracto colorido"],
                "accesorios": ["bolsa tote de algodón con ilustración o texto divertido"],
                "justificacion": "El tie-dye sobre pantalón de color es el estampado más alegre y sin género."
            },
            # Reemplaza: pantalón → cargo de color, camisa → camiseta gráfica
            "streetwear": {
                "prendas": ["pantalón cargo en mostaza, verde lima o coral", "camiseta gráfica de algodón con ilustración"],
                "accesorios": ["gorra bordada con motivos divertidos o de colores"],
                "justificacion": "El color en el streetwear rompe con lo oscuro y aporta mucha personalidad."
            },
            # Reemplaza: pantalón → recto en color sólido, camisa → básica neutra
            "minimal": {
                "prendas": ["pantalón recto de talle alto en color sólido vivo (amarillo, coral o azul eléctrico)", "camiseta básica en blanco o crema"],
                "accesorios": ["joyería de resina de color o dorada discreta"],
                "justificacion": "Un solo color potente con corte limpio: la alegría con criterio estético."
            },
            # Reemplaza todo: color en materiales inesperados
            "edgy": {
                "prendas": ["pantalón o falda de vinilo o charol en color inesperado (rosa chicle, verde lima)", "top negro básico de contraste"],
                "accesorios": ["botas de plataforma chunky en color que contraste"],
                "justificacion": "El color en materiales inesperados genera un happy subversivo muy interesante."
            },
        },

        "por_estacion": {
            "invierno": {
                "prendas": ["abrigo de paño en color vivo: rojo, amarillo mostaza o azul cobalto"],
                "accesorios": ["guantes de punto de colores mezclados o en contraste"],
                "justificacion": "Un abrigo de color en invierno es la declaración más alegre posible."
            },
            "verano": {
                "prendas": ["conjunto de lino o algodón coordinado en color pastel"],
                "accesorios": ["sombrero de paja de ala ancha con cinta de color"],
                "justificacion": "El conjunto coordinado en pastel es el look happy definitivo del verano."
            },
            "primavera": {
                "prendas": ["chaqueta vaquera teñida en tie-dye o bordada con motivos"],
                "accesorios": ["sneakers de plataforma en blanco o color pastel"],
                "justificacion": "La denim customizada en primavera es pura energía positiva."
            },
            "otoño": {
                "prendas": ["jersey de punto con motivos, letras o estampado jacquard colorido"],
                "accesorios": ["calcetines de colores visibles con mocasines o botas bajas"],
                "justificacion": "El jersey con motivos aporta humor y calidez al otoño gris."
            },
        },

        "por_clima": {
            "lluvia": {
                "prendas": ["impermeable en amarillo limón o naranja"],
                "accesorios": ["botas de agua de color con calcetines de rayas visibles"],
                "justificacion": "El impermeable de color convierte un día de lluvia en algo alegre."
            },
            "frio": {
                "prendas": ["jersey extra voluminoso con estampado jacquard colorido"],
                "accesorios": ["bufanda de rayas anchas en varios colores"],
                "justificacion": "El jacquard colorido es calidez y alegría en la misma prenda."
            },
            "calor": {
                "prendas": ["camiseta gráfica de algodón ligero con ilustración o slogan divertido"],
                "accesorios": ["gafas de sol de colores con cristal degradado o espejado"],
                "justificacion": "La camiseta gráfica es el vehículo más directo de la alegría en verano."
            },
        }
    },

    # ══════════════════════════════════════════════════════
    # CLUSTER 5 — FIESTA / SUBIDÓN
    # Base: top metalizado + pantalón tiro bajo
    # Para femenino: el vestido REEMPLAZA las 2 prendas base
    # ══════════════════════════════════════════════════════
    5: {
        "mood_name": "Fiesta / Subidón",

        "outfit_base": {
            "prendas": ["top de lentejuelas o tejido metalizado", "pantalón negro de tiro bajo o pitillo"],
            "accesorios": ["layering de collares finos o body chain"],
            "justificacion": "Brillo, actitud nocturna y silueta marcada: la combinación que define la fiesta."
        },

        "por_estilo": {
            # Reemplaza las 2 prendas base por el vestido (outfit completo)
            "femenino": {
                "prendas": ["minivestido de lentejuelas o tejido brillante", "chaqueta de punto o blazer fino encima para entrar"],
                "accesorios": ["tacones de aguja o sandalia de tiras con plataforma"],
                "justificacion": "El minivestido brillante con blazer fino es el look de fiesta femenino más icónico."
            },
            # Reemplaza: top → camisa satén, pantalón queda
            "masculino": {
                "prendas": ["camisa de satén o seda en color joya (burdeos, verde botella, azul cobalto)", "pantalón negro slim o pitillo"],
                "accesorios": ["mocasines de cuero con suela gruesa"],
                "justificacion": "La camisa de satén en color joya sobre pantalón negro es elegancia nocturna sin esfuerzo."
            },
            # Reemplaza: top → tirantes negro, pantalón → wide leg metalizado
            "unisex": {
                "prendas": ["top de tirantes negro básico", "pantalón wide leg plateado o dorado"],
                "accesorios": ["cinturón metálico ancho como accesorio protagonista"],
                "justificacion": "El wide leg metalizado es la pieza de fiesta más democrática y espectacular."
            },
            # Reemplaza todo: sporty pero brillante
            "streetwear": {
                "prendas": ["top llamativo de color o con logo grande", "pantalón de chándal satinado o iridiscente"],
                "accesorios": ["sneakers de plataforma con acabado reflectante"],
                "justificacion": "Streetwear llevado a la discoteca: sporty pero brillante y con actitud."
            },
            # Reemplaza todo: slip dress + blazer = fiesta elegante
            "minimal": {
                "prendas": ["vestido slip dress de satén en negro, dorado o champán", "blazer oversized encima para llegar"],
                "accesorios": ["pendientes de aro dorados XL como única joya"],
                "justificacion": "El slip dress con blazer es la versión más sofisticada y sin esfuerzo del look de fiesta."
            },
            # Reemplaza todo: cuero de arriba abajo
            "edgy": {
                "prendas": ["top de cuero o vinilo con recortes o asimétrico", "pantalón de cuero ajustado negro"],
                "accesorios": ["botas de plataforma hasta la rodilla en negro"],
                "justificacion": "Cuero de arriba abajo: la fiesta en su versión más oscura y dominante."
            },
        },

        "por_estacion": {
            "invierno": {
                "prendas": ["abrigo de pelo sintético en blanco, negro o rosa fucsia"],
                "accesorios": ["guantes de satén largos hasta el codo"],
                "justificacion": "El abrigo de pelo es el gran statement de las noches de invierno."
            },
            "verano": {
                "prendas": ["co-ord de lino o tela ligera metalizada para noche de verano"],
                "accesorios": ["gafas de sol espejadas para el after al amanecer"],
                "justificacion": "El co-ord metálico ligero es perfecto para fiestas de verano al aire libre."
            },
            "primavera": {
                "prendas": ["blazer de lentejuelas sobre look base sencillo"],
                "accesorios": ["sandalias de tiras finas con tacón fino"],
                "justificacion": "El blazer de lentejuelas transforma cualquier look básico en look de fiesta."
            },
            "otoño": {
                "prendas": ["chaqueta de cuero negro con detalles metálicos o bordados"],
                "accesorios": ["botines con tacón bloque y puntera fina"],
                "justificacion": "El cuero con detalles brillantes es la transición perfecta otoño-noche."
            },
        },

        "por_clima": {
            "lluvia": {
                "prendas": ["impermeable vinílico transparente sobre el outfit completo"],
                "accesorios": ["botas de agua con tacón y acabado brillante o negro mate"],
                "justificacion": "El PVC transparente protege el outfit sin taparlo: icónico en noches de lluvia."
            },
            "frio": {
                "prendas": ["body de manga larga con lentejuelas o brillantes bajo el pantalón"],
                "accesorios": ["abrigo largo de lana por encima del outfit de fiesta"],
                "justificacion": "El body de lentejuelas permite capear el frío sin sacrificar el brillo."
            },
            "calor": {
                "prendas": ["top bralette de lentejuelas o con strass como pieza única"],
                "accesorios": ["abanico de plumas o con lentejuelas"],
                "justificacion": "Menos tela, más brillo: la fórmula perfecta para fiesta con calor intenso."
            },
        }
    },

    # ══════════════════════════════════════════════════════
    # CLUSTER 6 — TRISTE / MELANCÓLICO
    # Base: pantalón de paño gris + jersey cuello barco
    # Para femenino: el vestido slip REEMPLAZA las prendas base
    # ══════════════════════════════════════════════════════
    6: {
        "mood_name": "Triste / Melancólico",

        "outfit_base": {
            "prendas": ["pantalón holgado de paño o lana en gris o azul pizarra", "jersey de punto flojo de cuello barco en gris claro"],
            "accesorios": ["bolso cruzado de cuero envejecido en marrón o negro"],
            "justificacion": "Tejidos suaves y siluetas envolventes en tonos fríos que acompañan sin oprimir."
        },

        "por_estilo": {
            # Reemplaza las 2 prendas por vestido slip (outfit completo) + prenda encima
            "femenino": {
                "prendas": ["vestido slip largo de satén en lavanda pálida o gris azulado", "cardigan fino oversize de mohair encima"],
                "accesorios": ["collar de perlas pequeñas o cuentas irregulares de cerámica"],
                "justificacion": "El slip largo con mohair encima tiene una belleza melancólica muy cinematográfica."
            },
            # Reemplaza: pantalón queda, jersey → camisa franela
            "masculino": {
                "prendas": ["pantalón holgado de paño en gris o azul marino", "camisa de franela oversize en cuadros apagados abierta sobre camiseta gris"],
                "accesorios": ["gorro de lana fino en gris o azul marino oscuro"],
                "justificacion": "La franela oversize abierta sobre gris es reconfort y estética en partes iguales."
            },
            # Reemplaza todo: abrigo oversized como pieza envolvente
            "unisex": {
                "prendas": ["camiseta gris de algodón pesado de manga larga", "pantalón holgado de punto en gris marengo", "abrigo de paño gris marengo oversized encima"],
                "accesorios": ["guantes de punto sin dedos en gris claro"],
                "justificacion": "El abrigo oversized en gris marengo envuelve como un abrazo y lo dice todo."
            },
            # Reemplaza todo: desgastado y desteñido urbano
            "streetwear": {
                "prendas": ["hoodie desteñido o con efecto aged en gris claro", "pantalón wide leg cargo en gris desgastado"],
                "accesorios": ["zapatillas vintage en blanco roto con suela amarillenta"],
                "justificacion": "Lo desgastado y desteñido encaja con la melancolía mejor que cualquier otra estética."
            },
            # Reemplaza todo: monocromático gris de corte preciso
            "minimal": {
                "prendas": ["pantalón de sastre gris perla de talle alto", "camiseta de canalé gris claro de manga larga slim"],
                "accesorios": ["anillo de plata oxidada o con piedra gris mate"],
                "justificacion": "El monocromático gris en piezas bien cortadas convierte la tristeza en elegancia contenida."
            },
            # Reemplaza todo: negro desgastado con historia
            "edgy": {
                "prendas": ["vestido negro largo asimétrico con detalle de rotos o dobladillo irregular", "chaqueta de cuero negro desgastada encima"],
                "accesorios": ["botas de combate negras hasta el tobillo con cordones gruesos"],
                "justificacion": "La melancolía en clave edgy: ropa que parece llevar años de historia vivida."
            },
        },

        "por_estacion": {
            "invierno": {
                "prendas": ["abrigo largo de lana en gris, azul noche o negro suave de corte recto"],
                "accesorios": ["bufanda de lana merino extrasuave en tono frío (gris, lavanda, azul niebla)"],
                "justificacion": "El abrigo largo oscuro en invierno amplifica la melancolía de forma hermosa."
            },
            "verano": {
                "prendas": ["vestido de algodón o lino holgado en blanco roto, azul niebla o gris claro"],
                "accesorios": ["sandalias planas de cuero natural con tira minimalista"],
                "justificacion": "La melancolía en verano es ligera y contemplativa, como una tarde nublada."
            },
            "primavera": {
                "prendas": ["cardigan de mohair en lavanda pálida, gris azulado o malva suave"],
                "accesorios": ["botines de piel fina con cordones en marrón oscuro o negro"],
                "justificacion": "El mohair en colores fríos tiene una textura que parece neblina primaveral."
            },
            "otoño": {
                "prendas": ["gabardina larga en beige grisáceo o camel apagado de corte recto"],
                "accesorios": ["gorro de fieltro o boina en gris carbón o azul oscuro"],
                "justificacion": "La gabardina con boina en otoño es la imagen más icónica de la melancolía urbana."
            },
        },

        "por_clima": {
            "lluvia": {
                "prendas": ["gabardina clásica de doble botonadura en beige o gris"],
                "accesorios": ["botas de agua en negro mate sin brillos"],
                "justificacion": "La lluvia y la gabardina son la combinación más melancólica que existe."
            },
            "frio": {
                "prendas": ["jersey de lana de punto grueso en gris marengo o azul medianoche"],
                "accesorios": ["bufanda oversize enrollada varias veces al cuello en tono frío"],
                "justificacion": "El jersey grueso oscuro es la armadura suave de los días melancólicos de frío."
            },
            "calor": {
                "prendas": ["camiseta de algodón con lavado desteñido en blanco roto o gris muy claro"],
                "accesorios": ["pulseras finas de hilo o plata apiladas en la muñeca"],
                "justificacion": "La camiseta lavada y desteñida tiene una belleza desvaída perfecta para la melancolía en calor."
            },
        }
    }
}
