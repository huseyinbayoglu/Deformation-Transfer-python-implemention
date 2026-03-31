Aşama 1: Temel Altyapı                                                                                                   
                                                                                                                           
  1. Mesh I/O — OBJ dosyalarını okuma/yazma (vertices, faces)                                                              
  2. Veri yapıları — Triangle mesh: vertices, faces, adjacency (hangi üçgen hangi üçgenle komşu), vertex-to-triangle       
  mapping (p(v))                                                                                                           
                                                                                                                         
  Aşama 2: Üçgen Başına Affine Dönüşüm Hesaplama (Section 3)                                                               
                                                                                                                         
  3. 4. vertex hesaplama — Her üçgen için normal yönünde sanal 4. vertex ekleme (Equation 1)                               
  4. V matrisi oluşturma — V = [v2-v1, v3-v1, v4-v1] (Equation 3)                                                        
  5. Q = Ṽ V⁻¹ hesaplama — Source mesh'in reference ve deformed halleri arasındaki affine dönüşümleri S1...S|S| bulma      
  (Equation 4)                                                                                                             
                                                                                                                           
  Aşama 3: Correspondence (Section 5) — En Zor Kısım                                                                       
                                                                                                                         
  6. Marker noktaları — Kullanıcının source ve target üzerinde eşleşen vertex'leri belirlemesi                             
  7. Smoothness terimi (ES) — Komşu üçgenlerin dönüşümlerinin benzer olması (Equation 11)
  8. Identity terimi (EI) — Dönüşümlerin identity'den çok sapmaması (Equation 12)                                          
  9. Closest point terimi (EC) — Her vertex'in target mesh üzerindeki en yakın geçerli noktaya çekilmesi (Equation 13)     
  10. İki fazlı optimizasyon — Önce EC=0 ile çöz, sonra wC'yi kademeli artırarak iteratif çöz (Equation 14)                
  11. Triangle correspondence çıkarma — Deformed source'un centroid'lerini target centroid'leriyle eşleştir                
                                                                                                                           
  Aşama 4: Deformation Transfer (Section 4) — Ana Algoritma                                                                
                                                                                                                           
  12. Vertex formülasyonu — Target dönüşümlerini T = Ṽ V⁻¹ ile vertex cinsinden yaz                                        
  13. Sparse lineer sistem kurma — min ||c - Ax̃||² (Equation 9)
  14. AᵀA faktorizasyonu — Sparse LU factorization (bir kez hesapla)                                                       
  15. Backsubstitution — Her yeni source pose için sadece geri-yerine-koyma ile çöz (x, y, z ayrı ayrı)                    
  16. Vertex constraints — Ayak sabitleme gibi pozisyon kısıtları                                                          
                                                                                                                           
  Aşama 5: Test & Görselleştirme                                                                                           
                                                                                                                           
  17. Basit test — Küçük mesh'lerle doğrulama                                                                              
  18. Görselleştirme — Matplotlib 3D veya Open3D ile sonuçları gösterme
  